/** streamChat SSE 解析单测：新契约帧格式 `event: <type>\ndata: {...}` + 旧格式兼容。 */
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'

import { streamChat, type SSEHandlers } from '@/composables/useSSE'

const TOKEN = 'test-token'

/** 构造 fetch mock 的 Response 壳：ok + body(ReadableStream) + text()。 */
function sseResponse(frames: string[]): { ok: boolean; body: ReadableStream<Uint8Array>; text: () => Promise<string> } {
  // 真实服务端每帧以空行结尾，最后补 \n\n 让解析器把末帧 flush 出来
  const bytes = new TextEncoder().encode(frames.join('\n\n') + '\n\n')
  return {
    ok: true,
    body: new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(bytes)
        controller.close()
      },
    }),
    text: async () => '',
  }
}

interface SpyHandlers {
  onStatus: ReturnType<typeof vi.fn>
  onAction: ReturnType<typeof vi.fn>
  onDone: ReturnType<typeof vi.fn>
  onError: ReturnType<typeof vi.fn>
}

async function runWith(frames: string[], overrides?: Partial<SSEHandlers>): Promise<SpyHandlers> {
  const onStatus = vi.fn()
  const onAction = vi.fn()
  const onDone = vi.fn()
  const onError = vi.fn()
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(sseResponse(frames)))
  await streamChat('sid-1', '你好', { onStatus, onAction, onDone, onError, ...overrides })
  return { onStatus, onAction, onDone, onError }
}

describe('streamChat', () => {
  beforeEach(() => {
    localStorage.clear()
    localStorage.setItem('cs_token', TOKEN)
    setActivePinia(createPinia())
    vi.unstubAllGlobals()
  })

  it('新契约帧格式 event: status\n data: {...} 触发 onStatus', async () => {
    const { onStatus } = await runWith([
      'event: status\ndata: {"type":"status","message":"正在查询订单"}',
    ])
    expect(onStatus).toHaveBeenCalledWith('正在查询订单')
  })

  it('done 事件携带 content 与 session_id', async () => {
    const { onDone } = await runWith([
      'event: done\ndata: {"type":"done","content":"退货申请已受理","session_id":"sid-9"}',
    ])
    expect(onDone).toHaveBeenCalledWith('退货申请已受理', 'sid-9')
  })

  it('action 事件传 intent；缺 intent 时为默认空串', async () => {
    const { onAction } = await runWith([
      'event: action\ndata: {"type":"action","action":"verify_order","intent":"RETURN_REQUEST"}',
      'event: action\ndata: {"type":"action","action":"refund"}',
    ])
    expect(onAction).toHaveBeenNthCalledWith(1, 'verify_order', 'RETURN_REQUEST')
    expect(onAction).toHaveBeenNthCalledWith(2, 'refund', '')
  })

  it('error 事件触发 onError；缺 message 用默认文案', async () => {
    const { onError } = await runWith([
      'event: error\ndata: {"type":"error","message":"订单不存在"}',
      'event: error\ndata: {"type":"error"}',
    ])
    expect(onError).toHaveBeenNthCalledWith(1, '订单不存在')
    expect(onError).toHaveBeenNthCalledWith(2, '系统出问题了，请稍后重试')
  })

  it('旧格式单行 data: {...} 兼容（status/done）', async () => {
    const { onStatus, onDone } = await runWith([
      'data: {"type":"status","message":"旧格式状态"}',
      'data: {"type":"done","content":"旧格式回复"}',
    ])
    expect(onStatus).toHaveBeenCalledWith('旧格式状态')
    expect(onDone).toHaveBeenCalledWith('旧格式回复', undefined)
  })

  it('损坏的 JSON 事件被忽略，不中断后续解析', async () => {
    const { onStatus, onError } = await runWith([
      'event: status\ndata: {"type":"status","message":"OK"}',
      'event: status\ndata: {broken-json',
      'event: done\ndata: {"type":"done","content":"收尾"}',
    ])
    expect(onStatus).toHaveBeenCalledWith('OK')
    expect(onError).not.toHaveBeenCalled()
  })

  it('请求头携带 Bearer token，body 为 JSON', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(sseResponse([])))
    await streamChat('sid-1', '你好', {})
    expect(fetch).toHaveBeenCalledWith('/api/v1/sessions/sid-1/messages', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${TOKEN}`,
      },
      body: JSON.stringify({ content: '你好' }),
    })
  })

  it('非 2xx 响应解析 detail 触发 onError', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: false,
        body: null,
        text: async () => '{"detail":"会话不存在"}',
      }),
    )
    const onError = vi.fn()
    await streamChat('sid-1', '你好', { onError })
    expect(onError).toHaveBeenCalledWith('会话不存在')
  })

  it('非 2xx 且 body 非 JSON 时用默认错误文案', async () => {
    vi.stubGlobal(
      'fetch',
      vi.fn().mockResolvedValue({
        ok: false,
        body: null,
        text: async () => '<html>502</html>',
      }),
    )
    const onError = vi.fn()
    await streamChat('sid-1', '你好', { onError })
    expect(onError).toHaveBeenCalledWith('请求失败，请稍后重试')
  })
})
