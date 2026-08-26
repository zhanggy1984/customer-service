/** SSE 流式接收：fetch POST + ReadableStream 解析（status/action/done/error）。 */
import { useAuthStore } from '@/stores/authStore'

export interface SSEHandlers {
  onStatus?: (message: string) => void
  onAction?: (action: string, intent: string) => void
  /** 思考链增量（reasoning 事件，先于 done 到达）。delta 为单帧增量，实时累积展示。 */
  onReasoning?: (delta: string) => void
  /** 工具调用事件（search_policy 检索结果透出，供回答下方来源标注展示）。 */
  onToolCall?: (name: string, result: unknown) => void
  onDone?: (reply: string, sessionId?: string) => void
  onError?: (message: string) => void
}

export async function streamChat(sid: string, content: string, handlers: SSEHandlers) {
  const auth = useAuthStore()
  const resp = await fetch(`/api/v1/sessions/${sid}/messages`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${auth.token}`,
    },
    body: JSON.stringify({ content }),
  })

  if (!resp.ok || !resp.body) {
    let detail = '请求失败，请稍后重试'
    try {
      const text = await resp.text()
      detail = JSON.parse(text).detail || detail
    } catch {
      /* 忽略 */
    }
    handlers.onError?.(detail)
    return
  }

  const reader = resp.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    const events = buffer.split('\n\n')
    buffer = events.pop() ?? ''
    for (const chunk of events) {
      // 兼容两种帧格式：旧 `data: {...}` 与契约新格式 `event: <type>\ndata: {...}`。
      // 逐行取第一条 data: 行，event: 帧头透明忽略。
      const dataLine = chunk.split('\n').find((l) => l.trim().startsWith('data:'))
      if (!dataLine) continue
      try {
        const evt = JSON.parse(dataLine.trim().slice(5))
        if (evt.type === 'status') handlers.onStatus?.(evt.message || '')
        else if (evt.type === 'action') handlers.onAction?.(evt.action, evt.intent || '')
        else if (evt.type === 'reasoning') handlers.onReasoning?.(evt.content || '')
        else if (evt.type === 'tool_call') handlers.onToolCall?.(evt.name || '', evt.result)
        else if (evt.type === 'done') handlers.onDone?.(evt.content || '', evt.session_id)
        else if (evt.type === 'error') handlers.onError?.(evt.message || '系统出问题了，请稍后重试')
      } catch {
        /* 忽略单个事件解析失败 */
      }
    }
  }
}
