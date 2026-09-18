// Tests for createRateLimitedTelegramApi — the wrapper that enforces
// per-chat FIFO ordering, a per-chat token bucket, a global token bucket,
// and 429 retry-after backoff on every outbound API call.
//
// The wrapper is transport-agnostic; we feed it a hand-rolled stub
// TelegramApi that records every call and can be programmed to throw
// grammY-shaped 429 errors. A fake clock + fake sleep give deterministic
// tests with no real wall-clock waits.

import { describe, expect, test } from 'bun:test'
import type {
  ChatAction,
  DownloadResult,
  EditOpts,
  SendDocumentOpts,
  SendMessageOpts,
  TelegramApi,
} from '../../src/channel/tools.js'
import type { Logger } from '../../src/log.js'
import {
  createFileFloodWaitStore,
  createJsonlRateLimitEventSink,
  createRateLimitedTelegramApi,
  TelegramFloodWaitError,
  type FloodWaitRecord,
  type FloodWaitStore,
  type RateLimitEvent,
  type RateLimitOptions,
} from '../../src/safety/rate-limited-telegram-api.js'
import { mkdtempSync, readFileSync, writeFileSync, existsSync } from 'fs'
import { tmpdir } from 'os'
import { join } from 'path'

interface SentCall {
  method:
    | 'sendMessage'
    | 'editMessageText'
    | 'setMessageReaction'
    | 'sendChatAction'
    | 'sendDocument'
    | 'sendPhoto'
    | 'deleteMessage'
    | 'downloadFile'
  chatId?: string
  messageId?: number
  text?: string
  emoji?: string
  action?: ChatAction
  filePath?: string
  fileId?: string
  opts?: SendMessageOpts | EditOpts | SendDocumentOpts
  ts: number
}

class FakeClock {
  // ms since start of test. Tests advance by calling `tick(ms)`.
  private t = 0
  now = (): number => this.t
  // Pending sleep resolvers, keyed by absolute wake time.
  private pending: Array<{ wakeAt: number; resolve: () => void }> = []
  sleep = (ms: number): Promise<void> => {
    if (ms <= 0) return Promise.resolve()
    return new Promise<void>((resolve) => {
      this.pending.push({ wakeAt: this.t + ms, resolve })
    })
  }
  async tick(ms: number): Promise<void> {
    this.t += ms
    // Resolve any sleeps whose wake time has passed. Resolve in order so
    // FIFO ordering is preserved.
    const due = this.pending
      .filter((p) => p.wakeAt <= this.t)
      .sort((a, b) => a.wakeAt - b.wakeAt)
    this.pending = this.pending.filter((p) => p.wakeAt > this.t)
    for (const p of due) p.resolve()
    // Yield to event loop so resolved promises propagate.
    await flushMicrotasks()
  }
}

// Drain microtask queue so awaiting code can advance.
function flushMicrotasks(): Promise<void> {
  return new Promise((resolve) => setImmediate(resolve))
}

// Build a grammY-shaped 429 error.
function make429Error(retryAfter: number | undefined): Error {
  const err = new Error('Too Many Requests') as Error & {
    error_code: number
    parameters: { retry_after?: number }
  }
  err.error_code = 429
  err.parameters = retryAfter !== undefined ? { retry_after: retryAfter } : {}
  return err
}

interface StubApi {
  api: TelegramApi
  calls: SentCall[]
  // Programmed errors per method/chat. When an error is set, the next
  // matching call throws it and the entry is consumed.
  queueError(method: SentCall['method'], err: Error): void
  // Times the method was entered, counting attempts that threw. `calls`
  // only records successes, so this is what proves a retry did (or did
  // not) happen.
  attempts(method: SentCall['method']): number
}

function makeStubApi(clock: FakeClock): StubApi {
  const calls: SentCall[] = []
  const errorQueue: Map<SentCall['method'], Error[]> = new Map()
  const attemptCount: Map<SentCall['method'], number> = new Map()
  const maybeThrow = (method: SentCall['method']): void => {
    attemptCount.set(method, (attemptCount.get(method) ?? 0) + 1)
    const list = errorQueue.get(method)
    if (list && list.length > 0) {
      const err = list.shift()
      if (err) throw err
    }
  }
  const api: TelegramApi = {
    async sendMessage(chatId, text, opts) {
      maybeThrow('sendMessage')
      calls.push({ method: 'sendMessage', chatId, text, opts, ts: clock.now() })
      return { message_id: calls.length }
    },
    async editMessageText(chatId, messageId, text, opts) {
      maybeThrow('editMessageText')
      calls.push({ method: 'editMessageText', chatId, messageId, text, opts, ts: clock.now() })
    },
    async setMessageReaction(chatId, messageId, emoji) {
      maybeThrow('setMessageReaction')
      calls.push({ method: 'setMessageReaction', chatId, messageId, emoji, ts: clock.now() })
    },
    async sendChatAction(chatId, action) {
      maybeThrow('sendChatAction')
      calls.push({ method: 'sendChatAction', chatId, action, ts: clock.now() })
    },
    async sendDocument(chatId, filePath, opts) {
      maybeThrow('sendDocument')
      calls.push({ method: 'sendDocument', chatId, filePath, opts, ts: clock.now() })
      return { message_id: calls.length }
    },
    async sendPhoto(chatId, filePath, opts) {
      maybeThrow('sendPhoto')
      calls.push({ method: 'sendPhoto', chatId, filePath, opts, ts: clock.now() })
      return { message_id: calls.length }
    },
    async downloadFile(fileId, _destDir) {
      maybeThrow('downloadFile')
      calls.push({ method: 'downloadFile', fileId, ts: clock.now() })
      return { path: '/tmp/x', size: 0 } satisfies DownloadResult
    },
    async deleteMessage(chatId, messageId) {
      maybeThrow('deleteMessage')
      calls.push({ method: 'deleteMessage', chatId, messageId, ts: clock.now() })
    },
  }
  return {
    api,
    calls,
    queueError(method, err) {
      const list = errorQueue.get(method) ?? []
      list.push(err)
      errorQueue.set(method, list)
    },
    attempts(method) {
      return attemptCount.get(method) ?? 0
    },
  }
}

const stubLog: Logger = {
  debug() {},
  info() {},
  warn() {},
  error() {},
}

function defaultOpts(clock: FakeClock): RateLimitOptions {
  return {
    perChatRefillPerSec: 1,
    perChatBurstCapacity: 3,
    globalRefillPerSec: 25,
    globalBurstCapacity: 25,
    maxRetries: 3,
    jitterMaxMs: 0, // deterministic
    now: clock.now,
    sleep: clock.sleep,
  }
}

describe('createRateLimitedTelegramApi — per-chat token bucket', () => {
  test('single send passes through immediately', async () => {
    const clock = new FakeClock()
    const stub = makeStubApi(clock)
    const api = createRateLimitedTelegramApi(stub.api, stubLog, defaultOpts(clock))
    const result = await api.sendMessage('100', 'hi', {})
    expect(result.message_id).toBe(1)
    expect(stub.calls.length).toBe(1)
    expect(stub.calls[0]?.ts).toBe(0)
  })

  test('first 3 sends to same chat consume burst capacity without waiting', async () => {
    const clock = new FakeClock()
    const stub = makeStubApi(clock)
    const api = createRateLimitedTelegramApi(stub.api, stubLog, defaultOpts(clock))
    await Promise.all([
      api.sendMessage('100', 'a', {}),
      api.sendMessage('100', 'b', {}),
      api.sendMessage('100', 'c', {}),
    ])
    expect(stub.calls.map((c) => c.text)).toEqual(['a', 'b', 'c'])
    expect(stub.calls.every((c) => c.ts === 0)).toBe(true)
  })

  test('4th send to same chat waits for refill (1s)', async () => {
    const clock = new FakeClock()
    const stub = makeStubApi(clock)
    const api = createRateLimitedTelegramApi(stub.api, stubLog, defaultOpts(clock))
    // Drain the burst.
    await api.sendMessage('100', 'a', {})
    await api.sendMessage('100', 'b', {})
    await api.sendMessage('100', 'c', {})
    expect(stub.calls.length).toBe(3)
    // Fourth send must wait ~1000ms for refill.
    const p = api.sendMessage('100', 'd', {})
    await flushMicrotasks()
    expect(stub.calls.length).toBe(3) // still queued
    await clock.tick(999)
    expect(stub.calls.length).toBe(3)
    await clock.tick(1)
    await p
    expect(stub.calls.length).toBe(4)
    expect(stub.calls[3]?.text).toBe('d')
  })

  test('sends to different chats run in parallel up to global cap', async () => {
    const clock = new FakeClock()
    const stub = makeStubApi(clock)
    const api = createRateLimitedTelegramApi(stub.api, stubLog, defaultOpts(clock))
    // 5 different chats — global cap is 25, per-chat each gets a fresh
    // bucket, so all 5 should fire at t=0.
    await Promise.all([
      api.sendMessage('100', 'a', {}),
      api.sendMessage('200', 'b', {}),
      api.sendMessage('300', 'c', {}),
      api.sendMessage('400', 'd', {}),
      api.sendMessage('500', 'e', {}),
    ])
    expect(stub.calls.length).toBe(5)
    expect(stub.calls.every((c) => c.ts === 0)).toBe(true)
  })

  test('FIFO order preserved within a single chat under concurrency', async () => {
    const clock = new FakeClock()
    const stub = makeStubApi(clock)
    const api = createRateLimitedTelegramApi(stub.api, stubLog, defaultOpts(clock))
    // Fire 5 sends concurrently — first 3 burst through, 4th waits 1s, 5th waits 2s.
    const p = Promise.all([
      api.sendMessage('100', '1', {}),
      api.sendMessage('100', '2', {}),
      api.sendMessage('100', '3', {}),
      api.sendMessage('100', '4', {}),
      api.sendMessage('100', '5', {}),
    ])
    await flushMicrotasks()
    expect(stub.calls.length).toBe(3)
    await clock.tick(1000)
    expect(stub.calls.length).toBe(4)
    await clock.tick(1000)
    await p
    expect(stub.calls.length).toBe(5)
    expect(stub.calls.map((c) => c.text)).toEqual(['1', '2', '3', '4', '5'])
  })
})

describe('createRateLimitedTelegramApi — global token bucket', () => {
  test('global cap caps parallel sends across chats', async () => {
    const clock = new FakeClock()
    const stub = makeStubApi(clock)
    const opts = defaultOpts(clock)
    opts.globalBurstCapacity = 2
    opts.globalRefillPerSec = 1
    const api = createRateLimitedTelegramApi(stub.api, stubLog, opts)
    // 3 different chats, each chat has fresh per-chat bucket. Global cap=2
    // means only 2 fire at t=0; the third waits 1s for global refill.
    const p = Promise.all([
      api.sendMessage('100', 'a', {}),
      api.sendMessage('200', 'b', {}),
      api.sendMessage('300', 'c', {}),
    ])
    await flushMicrotasks()
    expect(stub.calls.length).toBe(2)
    await clock.tick(1000)
    await p
    expect(stub.calls.length).toBe(3)
  })
})

describe('createRateLimitedTelegramApi — 429 retry-after', () => {
  test('429 with retry_after triggers single backoff and retries', async () => {
    const clock = new FakeClock()
    const stub = makeStubApi(clock)
    const api = createRateLimitedTelegramApi(stub.api, stubLog, defaultOpts(clock))
    stub.queueError('sendMessage', make429Error(2))
    const p = api.sendMessage('100', 'hi', {})
    await flushMicrotasks()
    // First call threw 429 — no recorded call yet (error path skips push).
    expect(stub.calls.length).toBe(0)
    // Advance < retry_after — still waiting.
    await clock.tick(1999)
    expect(stub.calls.length).toBe(0)
    await clock.tick(1)
    const r = await p
    expect(stub.calls.length).toBe(1)
    expect(r.message_id).toBe(1)
  })

  test('429 without retry_after falls back to 1s', async () => {
    const clock = new FakeClock()
    const stub = makeStubApi(clock)
    const api = createRateLimitedTelegramApi(stub.api, stubLog, defaultOpts(clock))
    stub.queueError('sendMessage', make429Error(undefined))
    const p = api.sendMessage('100', 'hi', {})
    await flushMicrotasks()
    await clock.tick(999)
    expect(stub.calls.length).toBe(0)
    await clock.tick(1)
    await p
    expect(stub.calls.length).toBe(1)
  })

  test('two consecutive 429s succeed on the third attempt', async () => {
    const clock = new FakeClock()
    const stub = makeStubApi(clock)
    const api = createRateLimitedTelegramApi(stub.api, stubLog, defaultOpts(clock))
    stub.queueError('sendMessage', make429Error(1))
    stub.queueError('sendMessage', make429Error(2))
    const p = api.sendMessage('100', 'hi', {})
    await flushMicrotasks()
    await clock.tick(1000) // first retry-after
    await flushMicrotasks()
    await clock.tick(2000) // second retry-after
    await p
    expect(stub.calls.length).toBe(1)
  })

  test('after maxRetries 429s, error propagates', async () => {
    const clock = new FakeClock()
    const stub = makeStubApi(clock)
    const opts = defaultOpts(clock)
    opts.maxRetries = 2
    const api = createRateLimitedTelegramApi(stub.api, stubLog, opts)
    stub.queueError('sendMessage', make429Error(1))
    stub.queueError('sendMessage', make429Error(1))
    const p = api.sendMessage('100', 'hi', {}).catch((e: unknown) => e)
    await flushMicrotasks()
    await clock.tick(1000)
    await flushMicrotasks()
    const result = await p
    expect(result).toBeInstanceOf(Error)
    expect((result as { error_code?: number }).error_code).toBe(429)
    expect(stub.calls.length).toBe(0)
  })

  test('non-429 errors propagate immediately without retry', async () => {
    const clock = new FakeClock()
    const stub = makeStubApi(clock)
    const api = createRateLimitedTelegramApi(stub.api, stubLog, defaultOpts(clock))
    const err = new Error('boom') as Error & { error_code?: number }
    err.error_code = 400
    stub.queueError('sendMessage', err)
    await expect(api.sendMessage('100', 'hi', {})).rejects.toMatchObject({ message: 'boom' })
    expect(stub.calls.length).toBe(0)
  })

  test('429 on editMessageText also retries (lighter bucket)', async () => {
    const clock = new FakeClock()
    const stub = makeStubApi(clock)
    const api = createRateLimitedTelegramApi(stub.api, stubLog, defaultOpts(clock))
    stub.queueError('editMessageText', make429Error(1))
    const p = api.editMessageText('100', 42, 'edited', {})
    await flushMicrotasks()
    await clock.tick(1000)
    await p
    expect(stub.calls.length).toBe(1)
  })
})

describe('createRateLimitedTelegramApi — retry_after clamp & edge values', () => {
  test('retry_after of exactly 60s still retries', async () => {
    const clock = new FakeClock()
    const stub = makeStubApi(clock)
    const api = createRateLimitedTelegramApi(stub.api, stubLog, defaultOpts(clock))
    stub.queueError('sendMessage', make429Error(60))
    const p = api.sendMessage('100', 'hi', {})
    await flushMicrotasks()
    await clock.tick(60_000)
    await p
    expect(stub.calls.length).toBe(1)
  })

  test('retry_after over 60s is a flood-wait: fails fast, never retries', async () => {
    const clock = new FakeClock()
    const stub = makeStubApi(clock)
    const api = createRateLimitedTelegramApi(stub.api, stubLog, defaultOpts(clock))
    stub.queueError('sendMessage', make429Error(30_710))
    const result = await api.sendMessage('100', 'hi', {}).catch((e: unknown) => e)
    // Rejected immediately — no sleep, no second request. Retrying inside a
    // flood-wait window re-arms the ban, which is the bug this guards.
    expect(result).toBeInstanceOf(TelegramFloodWaitError)
    expect(stub.calls.length).toBe(0)
    expect(stub.attempts('sendMessage')).toBe(1)
  })

  test('flood-wait error carries the true window and stays 429-shaped', async () => {
    const clock = new FakeClock()
    const stub = makeStubApi(clock)
    const api = createRateLimitedTelegramApi(stub.api, stubLog, defaultOpts(clock))
    const original = make429Error(30_710)
    stub.queueError('sendMessage', original)
    const result = (await api
      .sendMessage('100', 'hi', {})
      .catch((e: unknown) => e)) as TelegramFloodWaitError
    // Unclamped seconds, so the caller can schedule a resend...
    expect(result.retryAfterS).toBe(30_710)
    expect(result.windowOpensAtMs).toBe(clock.now() + 30_710_000)
    // ...and downstream `error_code === 429` checks keep working.
    expect(result.error_code).toBe(429)
    expect(result.cause).toBe(original)
  })

  test('breaker: after a flood-wait, later sends are rejected without hitting the API', async () => {
    const clock = new FakeClock()
    const stub = makeStubApi(clock)
    const api = createRateLimitedTelegramApi(stub.api, stubLog, defaultOpts(clock))
    stub.queueError('sendMessage', make429Error(30_710))
    const first = await api.sendMessage('100', 'first', {}).catch((e: unknown) => e)
    expect(first).toBeInstanceOf(TelegramFloodWaitError)
    expect(stub.attempts('sendMessage')).toBe(1)

    const second = await api.sendMessage('100', 'second', {}).catch((e: unknown) => e)
    expect(second).toBeInstanceOf(TelegramFloodWaitError)
    // The whole point: the API was never touched again, so nothing re-armed
    // the ban. Without this, each attempt bought another full window.
    expect(stub.attempts('sendMessage')).toBe(1)
    expect(stub.calls.length).toBe(0)
  })

  test('breaker reports the REMAINING window, not the original grant', async () => {
    const clock = new FakeClock()
    const stub = makeStubApi(clock)
    const api = createRateLimitedTelegramApi(stub.api, stubLog, defaultOpts(clock))
    const openedAt = clock.now()
    stub.queueError('sendMessage', make429Error(30_710))
    await api.sendMessage('100', 'first', {}).catch((e: unknown) => e)

    await clock.tick(10_000_000) // ~2h46m into the window
    const later = (await api
      .sendMessage('100', 'second', {})
      .catch((e: unknown) => e)) as TelegramFloodWaitError
    expect(later.retryAfterS).toBe(30_710 - 10_000)
    expect(later.windowOpensAtMs).toBe(openedAt + 30_710_000)
  })

  test('breaker closes once the window expires and sending resumes', async () => {
    const clock = new FakeClock()
    const stub = makeStubApi(clock)
    const api = createRateLimitedTelegramApi(stub.api, stubLog, defaultOpts(clock))
    stub.queueError('sendMessage', make429Error(30_710))
    await api.sendMessage('100', 'first', {}).catch((e: unknown) => e)

    await clock.tick(30_710_000)
    const resumed = await api.sendMessage('100', 'second', {})
    expect(resumed.message_id).toBe(1)
    expect(stub.calls.map((c) => c.text)).toEqual(['second'])
    expect(stub.attempts('sendMessage')).toBe(2)
  })

  test('breaker never shortens a window a later, smaller 429 would imply', async () => {
    const clock = new FakeClock()
    const stub = makeStubApi(clock)
    const api = createRateLimitedTelegramApi(stub.api, stubLog, defaultOpts(clock))
    stub.queueError('sendMessage', make429Error(30_710))
    await api.sendMessage('100', 'first', {}).catch((e: unknown) => e)
    const longWindow = clock.now() + 30_710_000

    // Window elapses; a fresh, much shorter flood-wait arrives.
    await clock.tick(30_710_000)
    stub.queueError('sendMessage', make429Error(120))
    const second = (await api
      .sendMessage('100', 'second', {})
      .catch((e: unknown) => e)) as TelegramFloodWaitError
    expect(second.windowOpensAtMs).toBe(clock.now() + 120_000)
    expect(second.windowOpensAtMs).toBeGreaterThan(longWindow)
  })

  test('retry_after = 0 is treated as 1s fallback', async () => {
    const clock = new FakeClock()
    const stub = makeStubApi(clock)
    const api = createRateLimitedTelegramApi(stub.api, stubLog, defaultOpts(clock))
    stub.queueError('sendMessage', make429Error(0))
    const p = api.sendMessage('100', 'hi', {})
    await flushMicrotasks()
    await clock.tick(1000)
    await p
    expect(stub.calls.length).toBe(1)
  })

  test('FIFO: second send waits for first retry to finish, then runs in order', async () => {
    const clock = new FakeClock()
    const stub = makeStubApi(clock)
    const api = createRateLimitedTelegramApi(stub.api, stubLog, defaultOpts(clock))
    // First call gets a 429 (retry_after=1), then succeeds.
    stub.queueError('sendMessage', make429Error(1))
    const p = Promise.all([
      api.sendMessage('100', 'first', {}),
      api.sendMessage('100', 'second', {}),
    ])
    await flushMicrotasks()
    // First call threw 429; second is still waiting on the chat tail.
    expect(stub.calls.length).toBe(0)
    await clock.tick(1000)
    await p
    expect(stub.calls.map((c) => c.text)).toEqual(['first', 'second'])
  })
})

describe('createRateLimitedTelegramApi — pass-through methods', () => {
  test('downloadFile is not rate-limited', async () => {
    const clock = new FakeClock()
    const stub = makeStubApi(clock)
    const api = createRateLimitedTelegramApi(stub.api, stubLog, defaultOpts(clock))
    // 10 parallel downloads should all fire at t=0.
    await Promise.all(
      Array.from({ length: 10 }, () => api.downloadFile('f', '/tmp')),
    )
    expect(stub.calls.length).toBe(10)
    expect(stub.calls.every((c) => c.ts === 0)).toBe(true)
  })

  test('editMessageText does not consume the per-chat send bucket', async () => {
    const clock = new FakeClock()
    const stub = makeStubApi(clock)
    const opts = defaultOpts(clock)
    opts.perChatBurstCapacity = 1
    const api = createRateLimitedTelegramApi(stub.api, stubLog, opts)
    // Use up the per-chat send bucket.
    await api.sendMessage('100', 'a', {})
    // Now do many edits — they must not be throttled by the send bucket.
    await Promise.all(
      Array.from({ length: 5 }, () => api.editMessageText('100', 42, 'edit', {})),
    )
    expect(stub.calls.filter((c) => c.method === 'editMessageText').length).toBe(5)
  })

  test('setMessageReaction is not gated by send bucket', async () => {
    const clock = new FakeClock()
    const stub = makeStubApi(clock)
    const opts = defaultOpts(clock)
    opts.perChatBurstCapacity = 1
    const api = createRateLimitedTelegramApi(stub.api, stubLog, opts)
    await api.sendMessage('100', 'a', {})
    await Promise.all(
      Array.from({ length: 5 }, () => api.setMessageReaction('100', 42, 'eyes')),
    )
    expect(stub.calls.filter((c) => c.method === 'setMessageReaction').length).toBe(5)
  })

  test('sendDocument and sendPhoto share the per-chat send bucket', async () => {
    const clock = new FakeClock()
    const stub = makeStubApi(clock)
    const opts = defaultOpts(clock)
    opts.perChatBurstCapacity = 2
    const api = createRateLimitedTelegramApi(stub.api, stubLog, opts)
    // Fire send + document + photo — third must wait since burst=2.
    const p = Promise.all([
      api.sendMessage('100', 'a', {}),
      api.sendDocument('100', '/tmp/d.pdf', {}),
      api.sendPhoto('100', '/tmp/p.jpg', {}),
    ])
    await flushMicrotasks()
    expect(stub.calls.length).toBe(2)
    await clock.tick(1000)
    await p
    expect(stub.calls.length).toBe(3)
  })
})

// ─────────────────────────────────────────────────────────────────────
// Flood-wait persistence, 429 journal, withFloodGuard (2026-09-18).
// Louis's bot: a restart inside a 56483 s window forgot the ban and the
// first reply re-armed it. These pin the fix.
// ─────────────────────────────────────────────────────────────────────

function memoryStore(initial: FloodWaitRecord | null = null): FloodWaitStore & {
  saved: FloodWaitRecord[]
  // Set to true to make save() report failure (disk full); the record is
  // still recorded in `attempts` so tests can count retries.
  failing: boolean
  attempts: FloodWaitRecord[]
} {
  const saved: FloodWaitRecord[] = []
  const attempts: FloodWaitRecord[] = []
  let current = initial
  const store = {
    saved,
    attempts,
    failing: false,
    load: () => current,
    save: (r: FloodWaitRecord): boolean => {
      attempts.push(r)
      if (store.failing) return false
      current = r
      saved.push(r)
      return true
    },
  }
  return store
}

describe('createRateLimitedTelegramApi — flood-wait survives restart', () => {
  test('a saved window still in the future is restored: first send suppressed, API untouched', async () => {
    const clock = new FakeClock()
    const stub = makeStubApi(clock)
    const store = memoryStore({
      until_ms: clock.now() + 50_000_000,
      method: 'sendMessage',
      retry_after_s: 56_483,
      seen_at: '2026-09-18T00:00:00.000Z',
    })
    const events: RateLimitEvent[] = []
    const api = createRateLimitedTelegramApi(stub.api, stubLog, {
      ...defaultOpts(clock),
      floodWaitStore: store,
      onRateLimitEvent: (e) => events.push(e),
    })
    const result = await api.sendMessage('100', 'after restart', {}).catch((e: unknown) => e)
    expect(result).toBeInstanceOf(TelegramFloodWaitError)
    expect((result as TelegramFloodWaitError).retryAfterS).toBe(50_000)
    expect(stub.attempts('sendMessage')).toBe(0)
    expect(events.map((e) => e.kind)).toEqual(['restored', 'suppressed'])
    expect(events[0]).toMatchObject({ kind: 'restored', method: 'sendMessage', retry_after_s: 50_000 })
  })

  test('an expired saved window is ignored and sending works', async () => {
    const clock = new FakeClock()
    clock.now = () => 100_000
    const stub = makeStubApi(clock)
    const store = memoryStore({ until_ms: 99_000, method: 'sendMessage', retry_after_s: 10, seen_at: '' })
    const api = createRateLimitedTelegramApi(stub.api, stubLog, { ...defaultOpts(clock), floodWaitStore: store })
    const sent = await api.sendMessage('100', 'ok', {})
    expect(sent.message_id).toBe(1)
    expect(store.saved.length).toBe(0)
  })

  test('a new flood-wait is saved with the method that earned it', async () => {
    const clock = new FakeClock()
    const stub = makeStubApi(clock)
    const store = memoryStore()
    const api = createRateLimitedTelegramApi(stub.api, stubLog, { ...defaultOpts(clock), floodWaitStore: store })
    stub.queueError('sendDocument', make429Error(30_710))
    await api.sendDocument('100', '/tmp/x.pdf', {}).catch(() => {})
    expect(store.saved.length).toBe(1)
    expect(store.saved[0]).toMatchObject({
      until_ms: clock.now() + 30_710_000,
      method: 'sendDocument',
      retry_after_s: 30_710,
    })
  })

  test('a fresh flood-wait after the old window expired is saved as a new record', async () => {
    const clock = new FakeClock()
    const stub = makeStubApi(clock)
    const store = memoryStore()
    const api = createRateLimitedTelegramApi(stub.api, stubLog, { ...defaultOpts(clock), floodWaitStore: store })
    stub.queueError('sendMessage', make429Error(30_710))
    await api.sendMessage('100', 'a', {}).catch(() => {})
    await clock.tick(30_710_000)
    stub.queueError('sendMessage', make429Error(120))
    await api.sendMessage('100', 'b', {}).catch(() => {})
    // Second window (120 s after expiry) IS longer than the expired one in
    // absolute terms, so it is saved — two records total, last one wins.
    expect(store.saved.length).toBe(2)
    expect(store.saved[1]?.retry_after_s).toBe(120)
  })

  test('a throwing event sink never breaks a send', async () => {
    const clock = new FakeClock()
    const stub = makeStubApi(clock)
    const api = createRateLimitedTelegramApi(stub.api, stubLog, {
      ...defaultOpts(clock),
      onRateLimitEvent: () => {
        throw new Error('sink broken')
      },
    })
    stub.queueError('sendMessage', make429Error(2))
    const p = api.sendMessage('100', 'hi', {})
    await flushMicrotasks()
    await clock.tick(2000)
    expect((await p).message_id).toBe(1)
  })
})

describe('createRateLimitedTelegramApi — 429 journal carries the method', () => {
  test('burst retry, flood-wait and suppression each emit one event with the real method and chat', async () => {
    const clock = new FakeClock()
    const stub = makeStubApi(clock)
    const events: RateLimitEvent[] = []
    const api = createRateLimitedTelegramApi(stub.api, stubLog, {
      ...defaultOpts(clock),
      onRateLimitEvent: (e) => events.push(e),
    })
    stub.queueError('sendPhoto', make429Error(3))
    const p = api.sendPhoto('7', '/tmp/a.jpg', {})
    await flushMicrotasks()
    await clock.tick(3000)
    await p
    stub.queueError('editMessageText', make429Error(30_710))
    await api.editMessageText('7', 1, 'x', {}).catch(() => {})
    await api.sendChatAction('8', 'typing').catch(() => {})
    expect(events).toEqual([
      { kind: 'burst_retry', method: 'sendPhoto', chat_id: '7', retry_after_s: 3, attempt: 1, wait_ms: 3000 },
      {
        kind: 'flood_wait',
        method: 'editMessageText',
        chat_id: '7',
        retry_after_s: 30_710,
        attempt: 1,
        window_opens_at: new Date(3000 + 30_710_000).toISOString(),
      },
      {
        kind: 'suppressed',
        method: 'sendChatAction',
        chat_id: '8',
        retry_after_s: 30_710,
        window_opens_at: new Date(3000 + 30_710_000).toISOString(),
        count: 1,
      },
    ])
  })
})

describe('createRateLimitedTelegramApi — withFloodGuard for calls outside TelegramApi', () => {
  test('runs the op and retries a burst 429 like any other method', async () => {
    const clock = new FakeClock()
    const stub = makeStubApi(clock)
    const events: RateLimitEvent[] = []
    const api = createRateLimitedTelegramApi(stub.api, stubLog, {
      ...defaultOpts(clock),
      onRateLimitEvent: (e) => events.push(e),
    })
    let n = 0
    const p = api.withFloodGuard('setMyCommands', async () => {
      n += 1
      if (n === 1) throw make429Error(5)
      return true
    })
    await flushMicrotasks()
    await clock.tick(5000)
    expect(await p).toBe(true)
    expect(n).toBe(2)
    expect(events[0]).toMatchObject({ kind: 'burst_retry', method: 'setMyCommands' })
    expect(events[0]).not.toHaveProperty('chat_id', expect.anything())
  })

  test('is suppressed inside a flood-wait without touching the API (the startup setMyCommands case)', async () => {
    const clock = new FakeClock()
    const stub = makeStubApi(clock)
    const api = createRateLimitedTelegramApi(stub.api, stubLog, {
      ...defaultOpts(clock),
      floodWaitStore: memoryStore({ until_ms: 10_000_000, method: 'sendMessage', retry_after_s: 10_000, seen_at: '' }),
    })
    let n = 0
    const result = await api
      .withFloodGuard('setMyCommands', async () => {
        n += 1
      })
      .catch((e: unknown) => e)
    expect(result).toBeInstanceOf(TelegramFloodWaitError)
    expect(n).toBe(0)
  })

  test('a flood-wait earned by withFloodGuard opens the breaker for ordinary sends too', async () => {
    const clock = new FakeClock()
    const stub = makeStubApi(clock)
    const api = createRateLimitedTelegramApi(stub.api, stubLog, defaultOpts(clock))
    await api.withFloodGuard('setMyCommands', async () => { throw make429Error(30_710) }).catch(() => {})
    const later = await api.sendMessage('100', 'x', {}).catch((e: unknown) => e)
    expect(later).toBeInstanceOf(TelegramFloodWaitError)
    expect(stub.attempts('sendMessage')).toBe(0)
  })
})

describe('createRateLimitedTelegramApi — hermes pre-merge checks (2026-09-18)', () => {
  test('guard runs again after a burst sleep: a window opened meanwhile stops the retry before it hits the API', async () => {
    const clock = new FakeClock()
    const stub = makeStubApi(clock)
    const api = createRateLimitedTelegramApi(stub.api, stubLog, defaultOpts(clock))
    // Chat A: burst 429, sleeps 5 s before retrying.
    stub.queueError('sendMessage', make429Error(5))
    const a = api.sendMessage('A', 'a', {}).catch((e: unknown) => e)
    await flushMicrotasks()
    expect(stub.attempts('sendMessage')).toBe(1)
    // Chat B, during A's sleep: flood-wait opens the breaker.
    stub.queueError('sendMessage', make429Error(30_710))
    await api.sendMessage('B', 'b', {}).catch(() => {})
    expect(stub.attempts('sendMessage')).toBe(2)
    // A wakes up: must NOT call the API again.
    await clock.tick(5000)
    const result = await a
    expect(result).toBeInstanceOf(TelegramFloodWaitError)
    expect(stub.attempts('sendMessage')).toBe(2)
  })

  test('queued sends behind a flood-wait are suppressed one by one without any API call', async () => {
    const clock = new FakeClock()
    const stub = makeStubApi(clock)
    const api = createRateLimitedTelegramApi(stub.api, stubLog, defaultOpts(clock))
    stub.queueError('sendMessage', make429Error(30_710))
    const results = await Promise.all(
      ['1', '2', '3'].map((t) => api.sendMessage('100', t, {}).catch((e: unknown) => e)),
    )
    expect(results.every((r) => r instanceof TelegramFloodWaitError)).toBe(true)
    expect(stub.attempts('sendMessage')).toBe(1)
  })

  test('a store whose load() throws does not stop the channel from starting and sending', async () => {
    const clock = new FakeClock()
    const stub = makeStubApi(clock)
    const api = createRateLimitedTelegramApi(stub.api, stubLog, {
      ...defaultOpts(clock),
      floodWaitStore: {
        load: () => {
          throw new Error('disk on fire')
        },
        save: () => true,
      },
    })
    expect((await api.sendMessage('100', 'ok', {})).message_id).toBe(1)
  })

  test('a store whose save() throws does not change the flood-wait outcome, and the window stays in memory', async () => {
    const clock = new FakeClock()
    const stub = makeStubApi(clock)
    const api = createRateLimitedTelegramApi(stub.api, stubLog, {
      ...defaultOpts(clock),
      floodWaitStore: {
        load: () => null,
        save: () => {
          throw new Error('EROFS')
        },
      },
    })
    stub.queueError('sendMessage', make429Error(30_710))
    const first = await api.sendMessage('100', 'a', {}).catch((e: unknown) => e)
    expect(first).toBeInstanceOf(TelegramFloodWaitError)
    const second = await api.sendMessage('100', 'b', {}).catch((e: unknown) => e)
    expect(second).toBeInstanceOf(TelegramFloodWaitError)
    expect(stub.attempts('sendMessage')).toBe(1)
  })

  test('a failed save is retried on later 429 events, at most once per 30 s, until it lands', async () => {
    const clock = new FakeClock()
    const stub = makeStubApi(clock)
    const store = memoryStore()
    store.failing = true
    const api = createRateLimitedTelegramApi(stub.api, stubLog, { ...defaultOpts(clock), floodWaitStore: store })
    stub.queueError('sendMessage', make429Error(30_710))
    await api.sendMessage('100', 'a', {}).catch(() => {})
    expect(store.attempts.length).toBe(1)
    expect(store.saved.length).toBe(0)
    // Immediate suppressed calls do not hammer the disk.
    await api.sendMessage('100', 'b', {}).catch(() => {})
    await api.sendMessage('100', 'c', {}).catch(() => {})
    expect(store.attempts.length).toBe(1)
    // 30 s later: one more attempt, still failing.
    await clock.tick(30_000)
    await api.sendMessage('100', 'd', {}).catch(() => {})
    expect(store.attempts.length).toBe(2)
    expect(store.saved.length).toBe(0)
    // Disk back: the next window passes and the ORIGINAL record lands.
    store.failing = false
    await clock.tick(30_000)
    await api.sendMessage('100', 'e', {}).catch(() => {})
    expect(store.saved.length).toBe(1)
    expect(store.saved[0]).toMatchObject({ until_ms: 30_710_000, method: 'sendMessage', retry_after_s: 30_710 })
    // Once persisted, nothing more is written.
    await clock.tick(30_000)
    await api.sendMessage('100', 'f', {}).catch(() => {})
    expect(store.attempts.length).toBe(3)
  })

  test('a restored window is not re-saved; a longer one seen later is', async () => {
    const clock = new FakeClock()
    const stub = makeStubApi(clock)
    const store = memoryStore({ until_ms: 1_000_000, method: 'sendMessage', retry_after_s: 1000, seen_at: '' })
    const api = createRateLimitedTelegramApi(stub.api, stubLog, { ...defaultOpts(clock), floodWaitStore: store })
    await api.sendMessage('100', 'a', {}).catch(() => {})
    expect(store.attempts.length).toBe(0)
    await clock.tick(1_000_000)
    stub.queueError('sendMessage', make429Error(5000))
    await api.sendMessage('100', 'b', {}).catch(() => {})
    expect(store.saved.length).toBe(1)
    expect(store.saved[0]?.until_ms).toBe(1_000_000 + 5_000_000)
  })

  test('suppressed calls are coalesced: one event per method per 30 s carrying the count', async () => {
    const clock = new FakeClock()
    const stub = makeStubApi(clock)
    const events: RateLimitEvent[] = []
    const api = createRateLimitedTelegramApi(stub.api, stubLog, {
      ...defaultOpts(clock),
      onRateLimitEvent: (e) => events.push(e),
    })
    stub.queueError('sendMessage', make429Error(30_710))
    await api.sendMessage('100', 'a', {}).catch(() => {})
    // Different chats: the per-chat bucket (burst 3) must not be what stops them.
    for (let i = 0; i < 5; i += 1) await api.sendMessage(`c${i}`, 'x', {}).catch(() => {})
    await api.sendChatAction('100', 'typing').catch(() => {})
    let suppressed = events.filter((e) => e.kind === 'suppressed')
    // First suppressed of each method reported at once, the other 4 sends counted.
    expect(suppressed.map((e) => [e.method, (e as { count: number }).count])).toEqual([
      ['sendMessage', 1],
      ['sendChatAction', 1],
    ])
    await clock.tick(30_000)
    await api.sendMessage('c9', 'y', {}).catch(() => {})
    suppressed = events.filter((e) => e.kind === 'suppressed')
    expect(suppressed.length).toBe(3)
    expect(suppressed[2]).toMatchObject({ kind: 'suppressed', method: 'sendMessage', count: 5 })
  })

  test('downloadFile (getFile) is suppressed inside a flood-wait and retries a burst 429', async () => {
    const clock = new FakeClock()
    const stub = makeStubApi(clock)
    const api = createRateLimitedTelegramApi(stub.api, stubLog, defaultOpts(clock))
    stub.queueError('downloadFile', make429Error(2))
    const p = api.downloadFile('f', '/tmp')
    await flushMicrotasks()
    await clock.tick(2000)
    expect((await p).path).toBe('/tmp/x')
    expect(stub.attempts('downloadFile')).toBe(2)
    stub.queueError('downloadFile', make429Error(30_710))
    await api.downloadFile('f', '/tmp').catch(() => {})
    const later = await api.downloadFile('f', '/tmp').catch((e: unknown) => e)
    expect(later).toBeInstanceOf(TelegramFloodWaitError)
    expect(stub.attempts('downloadFile')).toBe(3)
    // And a send is blocked by the window getFile earned.
    const send = await api.sendMessage('100', 'z', {}).catch((e: unknown) => e)
    expect(send).toBeInstanceOf(TelegramFloodWaitError)
    expect(stub.attempts('sendMessage')).toBe(0)
  })

  test('default clock is epoch milliseconds, so a persisted window compares across restarts', () => {
    const stub = makeStubApi(new FakeClock())
    const store = memoryStore({ until_ms: Date.now() + 60_000, method: 'sendMessage', retry_after_s: 60, seen_at: '' })
    const events: RateLimitEvent[] = []
    createRateLimitedTelegramApi(stub.api, stubLog, { floodWaitStore: store, onRateLimitEvent: (e) => events.push(e) })
    expect(events[0]?.kind).toBe('restored')
    expect((events[0] as { retry_after_s: number }).retry_after_s).toBeLessThanOrEqual(60)
  })
})

describe('createFileFloodWaitStore / createJsonlRateLimitEventSink — real files', () => {
  test('save then load round-trips; missing and corrupt files read as null', () => {
    const dir = mkdtempSync(join(tmpdir(), 'floodwait-'))
    const path = join(dir, 'nested', 'flood-wait.json')
    const store = createFileFloodWaitStore(path, stubLog)
    expect(store.load()).toBeNull()
    const rec: FloodWaitRecord = { until_ms: 123_456, method: 'sendMessage', retry_after_s: 60, seen_at: 't' }
    store.save(rec)
    expect(existsSync(path)).toBe(true)
    expect(existsSync(`${path}.tmp-${process.pid}`)).toBe(false)
    expect(store.load()).toEqual(rec)
    writeFileSync(path, '{not json')
    expect(store.load()).toBeNull()
    writeFileSync(path, JSON.stringify({ until_ms: 'soon' }))
    expect(store.load()).toBeNull()
  })

  test('save reports success, stamps bot_id, and a record from another bot is ignored on load', () => {
    const dir = mkdtempSync(join(tmpdir(), 'floodwait-'))
    const path = join(dir, 'flood-wait.json')
    const mine = createFileFloodWaitStore(path, stubLog, { botId: '111' })
    expect(mine.save({ until_ms: 5, method: 'sendMessage', retry_after_s: 1, seen_at: 't' })).toBe(true)
    expect(mine.load()).toMatchObject({ until_ms: 5, bot_id: '111' })
    const other = createFileFloodWaitStore(path, stubLog, { botId: '222' })
    expect(other.load()).toBeNull()
    // No bot id configured: the record is accepted whoever wrote it.
    expect(createFileFloodWaitStore(path, stubLog).load()).toMatchObject({ until_ms: 5 })
  })

  test('save returns false instead of throwing when the path cannot be written', () => {
    const dir = mkdtempSync(join(tmpdir(), 'floodwait-'))
    const blocker = join(dir, 'not-a-dir')
    writeFileSync(blocker, 'x')
    const store = createFileFloodWaitStore(join(blocker, 'flood-wait.json'), stubLog)
    expect(store.save({ until_ms: 5, method: 'sendMessage', retry_after_s: 1, seen_at: 't' })).toBe(false)
  })

  test('sink appends one JSON line per event with ts first and creates the directory', () => {
    const dir = mkdtempSync(join(tmpdir(), 'tg429-'))
    const path = join(dir, 'logs', 'telegram-429.jsonl')
    const sink = createJsonlRateLimitEventSink(path, stubLog)
    sink({ kind: 'suppressed', method: 'sendMessage', chat_id: '1', retry_after_s: 5, window_opens_at: 'w', count: 1 })
    sink({ kind: 'burst_retry', method: 'sendPhoto', retry_after_s: 2, attempt: 1, wait_ms: 2000 })
    const lines = readFileSync(path, 'utf8').trim().split('\n')
    expect(lines.length).toBe(2)
    const first = JSON.parse(lines[0] as string) as Record<string, unknown>
    expect(Object.keys(first)[0]).toBe('ts')
    expect(first).toMatchObject({ kind: 'suppressed', method: 'sendMessage', chat_id: '1' })
    expect(JSON.parse(lines[1] as string)).toMatchObject({ kind: 'burst_retry', method: 'sendPhoto' })
  })

  test('sink rotates the journal once to .1 past 5 MB', () => {
    const dir = mkdtempSync(join(tmpdir(), 'tg429-'))
    const path = join(dir, 'telegram-429.jsonl')
    writeFileSync(path, 'x'.repeat(5 * 1024 * 1024))
    const sink = createJsonlRateLimitEventSink(path, stubLog)
    sink({ kind: 'restored', method: 'sendMessage', retry_after_s: 1, window_opens_at: 'w' })
    expect(existsSync(`${path}.1`)).toBe(true)
    const lines = readFileSync(path, 'utf8').trim().split('\n')
    expect(lines.length).toBe(1)
    expect(JSON.parse(lines[0] as string)).toMatchObject({ kind: 'restored' })
  })
})
