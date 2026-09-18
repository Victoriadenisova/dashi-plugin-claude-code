// Outbound rate-limit wrapper around TelegramApi.
//
// Goal: make Telegram's per-chat / per-bot rate limits invisible to callers.
// A burst of replies (e.g. a multi-part report) used to surface as a 429
// from Bot API with retry_after ≈ 300s — long enough that the warchief lost
// sight of what the agent was doing. This wrapper enforces pacing BEFORE
// the request leaves the process and transparently retries on 429.
//
// Layers (independent, all consulted on every text-send):
//   1. Per-chat token bucket (default: 1 msg/sec sustained, burst 3).
//      Same-chat ordering is preserved via a FIFO tail-promise chain — a
//      second sendMessage to the same chat awaits the first before checking
//      its bucket. Different chats run in parallel.
//   2. Global token bucket (default: 25 msg/sec, burst 25). Caps total
//      throughput across all chats under Telegram's 30/sec bot-wide limit.
//   3. 429 handling: on a grammY-shaped 429 (`error_code: 429`, optional
//      `parameters.retry_after`), a SHORT wait (<= MAX_RETRY_AFTER_S) is
//      slept off with a small jitter and the SAME call is retried, bounded
//      by `maxRetries` (default 3). A LONG wait is a bot-wide flood-wait
//      and is never retried — it throws TelegramFloodWaitError with the
//      real window. See MAX_RETRY_AFTER_S for why retrying makes it worse.
//   4. Flood-wait breaker: after a long wait is seen, every call is
//      rejected locally until the window expires, so no request can re-arm
//      the ban. See `floodWaitUntilMs`. The window is persisted through
//      `opts.floodWaitStore` so a process restart inside it does not forget
//      the ban and re-arm it with the first reply (Louis, 2026-09-18: 56483 s).
//   5. Every 429 (burst retry, flood-wait, local suppression) is reported
//      through `opts.onRateLimitEvent` with the Telegram METHOD name, so the
//      first call that earned a ban can be found afterwards instead of
//      guessed. server.ts wires this to logs/telegram-429.jsonl.
//
// Calls that are not part of TelegramApi (e.g. `setMyCommands` at startup)
// go through `withFloodGuard(method, op)` on the returned object so they get
// the same breaker + 429 handling instead of bypassing it.
//
// Methods that don't consume the send-bucket: editMessageText (Telegram's
// edit limits are far more lenient), setMessageReaction, sendChatAction,
// deleteMessage, downloadFile. They still get the 429 retry wrapper so a
// stray 429 on an edit can recover without the caller seeing it.
//
// Test seams: `opts.now` and `opts.sleep` replace the real clock and
// setTimeout-based sleep, so tests can run instantly with deterministic
// virtual time.

import { appendFileSync, mkdirSync, readFileSync, renameSync, writeFileSync } from 'fs'
import { dirname } from 'path'
import type { Logger } from '../log.js'
import type {
  ChatAction,
  DownloadResult,
  EditOpts,
  SendDocumentOpts,
  SendMessageOpts,
  TelegramApi,
} from '../channel/tools.js'

interface TokenBucket {
  tokens: number
  capacity: number
  refillPerMs: number
  lastRefill: number
}

function makeBucket(capacity: number, refillPerSec: number, now: number): TokenBucket {
  return {
    tokens: capacity,
    capacity,
    refillPerMs: refillPerSec / 1000,
    lastRefill: now,
  }
}

function refill(b: TokenBucket, now: number): void {
  const dt = now - b.lastRefill
  if (dt <= 0) return
  b.tokens = Math.min(b.capacity, b.tokens + dt * b.refillPerMs)
  b.lastRefill = now
}

// ms to wait before consuming one token; 0 means available now.
function waitMs(b: TokenBucket, now: number): number {
  refill(b, now)
  if (b.tokens >= 1) return 0
  return Math.ceil((1 - b.tokens) / b.refillPerMs)
}

function consume(b: TokenBucket): void {
  b.tokens -= 1
}

interface ChatState {
  bucket: TokenBucket
  // FIFO tail: every enqueued op awaits this before checking the bucket.
  // Replaced with a fresh deferred at each enqueue. Errors do NOT propagate
  // (we use `.catch(() => {})` on the await) so one failed send cannot
  // permanently break the chain for a chat.
  //
  // HEAD-OF-LINE BLOCKING: a slow op (e.g. a 429 retry holding the lock)
  // delays all subsequent sends to the SAME chat. That's intentional —
  // ordering matters more than throughput for a conversational channel.
  // Different chats run in parallel (separate ChatState entries) so a stuck
  // chat does not affect others. Worst-case per-chat stall is bounded by
  // `maxRetries * MAX_RETRY_AFTER_S`.
  tail: Promise<void>
}

export interface RateLimitOptions {
  perChatRefillPerSec?: number
  perChatBurstCapacity?: number
  globalRefillPerSec?: number
  globalBurstCapacity?: number
  maxRetries?: number
  jitterMaxMs?: number
  /** Test seam: replace Date.now() for deterministic virtual time. */
  now?: () => number
  /** Test seam: replace setTimeout-based sleep. */
  sleep?: (ms: number) => Promise<void>
  /**
   * Persists the flood-wait window across restarts. Loaded once at
   * construction; saved whenever a new (or longer) window is learned.
   * Omit for an in-memory breaker only (tests, ad-hoc tooling).
   */
  floodWaitStore?: FloodWaitStore
  /** Receives one event per 429-related decision. Must not throw. */
  onRateLimitEvent?: (event: RateLimitEvent) => void
}

/** What we remember about a flood-wait: enough to restore the breaker and to
 *  say afterwards which call earned it. */
export interface FloodWaitRecord {
  /** Epoch ms when the window closes. */
  until_ms: number
  /** Telegram method whose 429 opened (or extended) the window. */
  method: string
  retry_after_s: number
  /** ISO timestamp of the 429. */
  seen_at: string
}

export interface FloodWaitStore {
  /** Returns the last saved record, or null when none / unreadable. */
  load(): FloodWaitRecord | null
  save(record: FloodWaitRecord): void
}

export type RateLimitEvent =
  | {
      kind: 'burst_retry'
      method: string
      chat_id?: string | undefined
      retry_after_s: number
      attempt: number
      wait_ms: number
    }
  | {
      kind: 'flood_wait'
      method: string
      chat_id?: string | undefined
      retry_after_s: number
      attempt: number
      window_opens_at: string
    }
  | {
      kind: 'suppressed'
      method: string
      chat_id?: string | undefined
      retry_after_s: number
      window_opens_at: string
    }
  | {
      kind: 'restored'
      method: string
      retry_after_s: number
      window_opens_at: string
    }

export interface RateLimitedTelegramApi extends TelegramApi {
  /**
   * Run an arbitrary Bot API call under the flood-wait breaker and the 429
   * retry policy, without the per-chat send bucket. For calls that are not
   * on TelegramApi (startup `setMyCommands`, future one-offs). `method` is
   * the Telegram method name and ends up in the 429 log.
   */
  withFloodGuard<T>(method: string, op: () => Promise<T>): Promise<T>
}

/**
 * File-backed FloodWaitStore. One small JSON file, written atomically
 * (tmp + rename) so a crash mid-write cannot leave a half record. A missing
 * or corrupt file reads as "no flood-wait known" — never as an error, the
 * channel must start regardless.
 */
export function createFileFloodWaitStore(path: string, log: Logger): FloodWaitStore {
  return {
    load(): FloodWaitRecord | null {
      let raw: string
      try {
        raw = readFileSync(path, 'utf8')
      } catch {
        return null
      }
      try {
        const v = JSON.parse(raw) as Partial<FloodWaitRecord>
        if (
          typeof v !== 'object' ||
          v === null ||
          typeof v.until_ms !== 'number' ||
          !Number.isFinite(v.until_ms) ||
          typeof v.method !== 'string'
        ) {
          log.warn('flood-wait state file malformed, ignoring', { path })
          return null
        }
        return {
          until_ms: v.until_ms,
          method: v.method,
          retry_after_s: typeof v.retry_after_s === 'number' ? v.retry_after_s : 0,
          seen_at: typeof v.seen_at === 'string' ? v.seen_at : '',
        }
      } catch (err) {
        log.warn('flood-wait state file unreadable, ignoring', {
          path,
          error: err instanceof Error ? err.message : String(err),
        })
        return null
      }
    },
    save(record: FloodWaitRecord): void {
      try {
        mkdirSync(dirname(path), { recursive: true })
        const tmp = `${path}.tmp-${process.pid}`
        writeFileSync(tmp, JSON.stringify(record) + '\n', { mode: 0o600 })
        renameSync(tmp, path)
      } catch (err) {
        // Losing persistence is bad but must not turn a 429 into a crash.
        log.warn('flood-wait state file write failed', {
          path,
          error: err instanceof Error ? err.message : String(err),
        })
      }
    },
  }
}

/**
 * JSONL sink for RateLimitEvent: one line per event, `ts` first. Append-only
 * so a ban can be traced back to the exact method and time afterwards.
 */
export function createJsonlRateLimitEventSink(
  path: string,
  log: Logger,
): (event: RateLimitEvent) => void {
  let dirReady = false
  return (event: RateLimitEvent): void => {
    try {
      if (!dirReady) {
        mkdirSync(dirname(path), { recursive: true })
        dirReady = true
      }
      appendFileSync(path, JSON.stringify({ ts: new Date().toISOString(), ...event }) + '\n', {
        mode: 0o600,
      })
    } catch (err) {
      log.warn('telegram 429 log write failed', {
        path,
        error: err instanceof Error ? err.message : String(err),
      })
    }
  }
}

interface Grammy429 {
  error_code: 429
  parameters?: { retry_after?: number }
}

// Telegram's 429 comes in two flavours that need opposite handling:
//
//   • Burst hiccup — `retry_after` of a few seconds, caused by our own
//     pacing. Sleeping it off and retrying is correct and stays invisible
//     to the caller.
//   • Flood-wait — `retry_after` of minutes to HOURS, imposed on the bot
//     itself. Requests sent inside that window are not queued, they RE-ARM
//     the ban: on 2026-08-20 a 20191 s wait on richard's bot became
//     30710 s after two automatic retries — the retries alone bought ~3
//     extra hours of silence, and the message was dropped anyway.
//
// MAX_RETRY_AFTER_S is the line between the two. At or below it we retry
// as before. Above it we do NOT touch the API again: we fail fast and hand
// the caller a TelegramFloodWaitError carrying the true, unclamped wake-up
// time so it can resend once the window actually opens.
//
// This also keeps the per-chat FIFO tail short. The tail blocks every
// later send to the same chat until the in-flight op finishes, so the
// worst-case stall stays maxRetries × MAX_RETRY_AFTER_S (3 × 60s = 3 min).
const MAX_RETRY_AFTER_S = 60

/**
 * Thrown instead of retrying when Telegram reports a flood-wait longer than
 * MAX_RETRY_AFTER_S. `error_code` stays 429 so existing 429 checks keep
 * firing; `retryAfterS` / `windowOpensAtMs` carry the real window so the
 * caller can schedule a resend instead of guessing.
 */
export class TelegramFloodWaitError extends Error {
  readonly error_code = 429
  readonly retryAfterS: number
  readonly windowOpensAtMs: number

  constructor(method: string, retryAfterS: number, windowOpensAtMs: number, cause: unknown) {
    super(
      `Telegram flood-wait on ${method}: ${retryAfterS}s remaining, window opens ` +
        `${new Date(windowOpensAtMs).toISOString()}. Not retried — a retry inside ` +
        `the window extends the ban. Resend after that time.`,
      { cause },
    )
    this.name = 'TelegramFloodWaitError'
    this.retryAfterS = retryAfterS
    this.windowOpensAtMs = windowOpensAtMs
  }
}

function parse429(err: unknown): { retryAfter: number } | null {
  if (typeof err !== 'object' || err === null) return null
  const e = err as Grammy429
  if (e.error_code !== 429) return null
  const after = e.parameters?.retry_after
  // Telegram's retry_after is in seconds. Coerce to a sane positive integer.
  // Deliberately NOT clamped here: withRetry needs the true value to tell a
  // burst hiccup from a flood-wait, and to report the real window.
  if (typeof after !== 'number' || !Number.isFinite(after) || after < 1) {
    return { retryAfter: 1 }
  }
  return { retryAfter: Math.ceil(after) }
}

export function createRateLimitedTelegramApi(
  raw: TelegramApi,
  log: Logger,
  opts: RateLimitOptions = {},
): RateLimitedTelegramApi {
  const cfg = {
    perChatRefillPerSec: opts.perChatRefillPerSec ?? 1,
    perChatBurstCapacity: opts.perChatBurstCapacity ?? 3,
    globalRefillPerSec: opts.globalRefillPerSec ?? 25,
    globalBurstCapacity: opts.globalBurstCapacity ?? 25,
    maxRetries: opts.maxRetries ?? 3,
    jitterMaxMs: opts.jitterMaxMs ?? 150,
  }
  const now = opts.now ?? ((): number => Date.now())
  const sleep =
    opts.sleep ??
    ((ms: number): Promise<void> =>
      ms <= 0 ? Promise.resolve() : new Promise((r) => setTimeout(r, ms)))

  const globalBucket = makeBucket(cfg.globalBurstCapacity, cfg.globalRefillPerSec, now())
  const chatState = new Map<string, ChatState>()

  // Circuit breaker for a bot-wide flood-wait. While one is in force every
  // further request RE-ARMS it rather than queueing behind it: on richard's
  // bot (2026-08-20) five reply attempts spread over 15 minutes each got the
  // identical `retry_after: 30710` back, so the window never came closer and
  // the agent would have stayed mute indefinitely. Once we learn a flood-wait
  // exists we therefore stop talking to Telegram altogether until it expires
  // and reject locally instead. That silence is what lets the ban run out.
  // 0 = no flood-wait known.
  let floodWaitUntilMs = 0
  const store = opts.floodWaitStore
  const emit = (event: RateLimitEvent): void => {
    try {
      opts.onRateLimitEvent?.(event)
    } catch (err) {
      log.warn('rate-limit event sink threw', {
        error: err instanceof Error ? err.message : String(err),
      })
    }
  }

  // Restore a window that outlived the previous process. Without this a
  // restart inside the ban forgets it, and the very next reply re-arms the
  // full window. An expired record is ignored (and left on disk; the next
  // flood-wait overwrites it).
  if (store) {
    const saved = store.load()
    if (saved && saved.until_ms > now()) {
      floodWaitUntilMs = saved.until_ms
      const remainingS = Math.ceil((saved.until_ms - now()) / 1000)
      log.warn('telegram flood-wait restored from state, suppressing sends', {
        method: saved.method,
        retry_after_s: remainingS,
        window_opens_at: new Date(saved.until_ms).toISOString(),
        seen_at: saved.seen_at,
      })
      emit({
        kind: 'restored',
        method: saved.method,
        retry_after_s: remainingS,
        window_opens_at: new Date(saved.until_ms).toISOString(),
      })
    }
  }

  function getChatState(chatId: string): ChatState {
    let s = chatState.get(chatId)
    if (!s) {
      s = {
        bucket: makeBucket(cfg.perChatBurstCapacity, cfg.perChatRefillPerSec, now()),
        tail: Promise.resolve(),
      }
      chatState.set(chatId, s)
    }
    return s
  }

  // Wait until both the per-chat and global buckets have a token, then
  // consume one from each. Caller is responsible for holding the per-chat
  // FIFO lock so two enqueues for the same chat can't race this check.
  async function waitForCapacity(state: ChatState): Promise<void> {
    // Loop because after waking from sleep, another global consumer may
    // have stolen the token we expected. Recompute and re-sleep if so.
    // In single-threaded JS this is rare but the loop keeps invariants
    // robust against future async interleaving.
    for (;;) {
      const t = now()
      const chatWait = waitMs(state.bucket, t)
      const globalWait = waitMs(globalBucket, t)
      const w = Math.max(chatWait, globalWait)
      if (w === 0) {
        consume(state.bucket)
        consume(globalBucket)
        return
      }
      await sleep(w)
    }
  }

  // `maxRetries` is the MAX NUMBER OF ATTEMPTS including the initial call.
  // Semantically: budget of how many times we hit Telegram for this op.
  // maxRetries=3 → up to 3 attempts (2 retries after the first failure).
  async function withRetry<T>(
    method: string,
    op: () => Promise<T>,
    chatId?: string,
  ): Promise<T> {
    let attempt = 0
    let lastErr: unknown
    while (true) {
      attempt += 1
      const suppressedForMs = floodWaitUntilMs - now()
      if (suppressedForMs > 0) {
        // Breaker open — do not touch the API, it would only re-arm the ban.
        const retryAfterS = Math.ceil(suppressedForMs / 1000)
        const windowOpensAt = new Date(floodWaitUntilMs).toISOString()
        log.warn('telegram flood-wait in force, request suppressed', {
          method,
          retry_after_s: retryAfterS,
          window_opens_at: windowOpensAt,
        })
        emit({ kind: 'suppressed', method, chat_id: chatId, retry_after_s: retryAfterS, window_opens_at: windowOpensAt })
        throw new TelegramFloodWaitError(method, retryAfterS, floodWaitUntilMs, lastErr)
      }
      try {
        return await op()
      } catch (err) {
        const r = parse429(err)
        if (r === null) throw err
        lastErr = err
        if (r.retryAfter > MAX_RETRY_AFTER_S) {
          // Flood-wait, not a burst. Stop here — see MAX_RETRY_AFTER_S — and
          // open the breaker so nothing else re-arms it. Never shorten a
          // window we already know about.
          const windowOpensAtMs = now() + r.retryAfter * 1000
          const extended = windowOpensAtMs > floodWaitUntilMs
          floodWaitUntilMs = Math.max(floodWaitUntilMs, windowOpensAtMs)
          const windowOpensAt = new Date(windowOpensAtMs).toISOString()
          log.warn('telegram flood-wait, not retrying', {
            method,
            retry_after_s: r.retryAfter,
            attempt,
            window_opens_at: windowOpensAt,
          })
          emit({ kind: 'flood_wait', method, chat_id: chatId, retry_after_s: r.retryAfter, attempt, window_opens_at: windowOpensAt })
          if (extended && store) {
            store.save({
              until_ms: floodWaitUntilMs,
              method,
              retry_after_s: r.retryAfter,
              seen_at: new Date(now()).toISOString(),
            })
          }
          throw new TelegramFloodWaitError(method, r.retryAfter, floodWaitUntilMs, err)
        }
        if (attempt >= cfg.maxRetries) break
        const jitter =
          cfg.jitterMaxMs > 0 ? Math.floor(Math.random() * cfg.jitterMaxMs) : 0
        const waitTotalMs = r.retryAfter * 1000 + jitter
        log.warn('telegram 429, backing off', {
          method,
          retry_after_s: r.retryAfter,
          attempt,
          wait_ms: waitTotalMs,
        })
        emit({ kind: 'burst_retry', method, chat_id: chatId, retry_after_s: r.retryAfter, attempt, wait_ms: waitTotalMs })
        await sleep(waitTotalMs)
      }
    }
    throw lastErr
  }

  // Serialize per-chat outbound work: each new op awaits the previous op
  // (without inheriting its error), then runs under the rate-limit gate.
  async function enqueueSend<T>(
    chatId: string,
    method: string,
    op: () => Promise<T>,
  ): Promise<T> {
    const state = getChatState(chatId)
    const prev = state.tail
    let release!: () => void
    state.tail = new Promise<void>((r) => {
      release = r
    })
    try {
      await prev.catch(() => {})
      await waitForCapacity(state)
      return await withRetry(method, op, chatId)
    } finally {
      release()
    }
  }

  return {
    async sendMessage(
      chatId: string,
      text: string,
      sendOpts: SendMessageOpts,
    ): Promise<{ message_id: number }> {
      return enqueueSend(chatId, 'sendMessage', () => raw.sendMessage(chatId, text, sendOpts))
    },

    async editMessageText(
      chatId: string,
      messageId: number,
      text: string,
      editOpts: EditOpts,
    ): Promise<void> {
      return withRetry(
        'editMessageText',
        () => raw.editMessageText(chatId, messageId, text, editOpts),
        chatId,
      )
    },

    async setMessageReaction(
      chatId: string,
      messageId: number,
      emoji: string,
    ): Promise<void> {
      return withRetry(
        'setMessageReaction',
        () => raw.setMessageReaction(chatId, messageId, emoji),
        chatId,
      )
    },

    async sendChatAction(chatId: string, action: ChatAction): Promise<void> {
      return withRetry('sendChatAction', () => raw.sendChatAction(chatId, action), chatId)
    },

    async sendDocument(
      chatId: string,
      filePath: string,
      docOpts: SendDocumentOpts,
    ): Promise<{ message_id: number }> {
      return enqueueSend(chatId, 'sendDocument', () => raw.sendDocument(chatId, filePath, docOpts))
    },

    async sendPhoto(
      chatId: string,
      filePath: string,
      photoOpts: SendDocumentOpts,
    ): Promise<{ message_id: number }> {
      return enqueueSend(chatId, 'sendPhoto', () => raw.sendPhoto(chatId, filePath, photoOpts))
    },

    async downloadFile(fileId: string, destDir: string): Promise<DownloadResult> {
      return raw.downloadFile(fileId, destDir)
    },

    async deleteMessage(chatId: string, messageId: number): Promise<void> {
      return withRetry('deleteMessage', () => raw.deleteMessage(chatId, messageId), chatId)
    },

    async withFloodGuard<T>(method: string, op: () => Promise<T>): Promise<T> {
      return withRetry(method, op)
    },
  }
}
