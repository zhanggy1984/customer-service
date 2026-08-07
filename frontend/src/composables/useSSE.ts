/** SSE 流式接收：fetch POST + ReadableStream 解析（status/action/done/error）。 */
import { useAuthStore } from '@/stores/authStore'

export interface SSEHandlers {
  onStatus?: (message: string) => void
  onAction?: (action: string, intent: string) => void
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
      const line = chunk.trim()
      if (!line.startsWith('data:')) continue
      try {
        const evt = JSON.parse(line.slice(5))
        if (evt.type === 'status') handlers.onStatus?.(evt.message || '')
        else if (evt.type === 'action') handlers.onAction?.(evt.action, evt.intent || '')
        else if (evt.type === 'done') handlers.onDone?.(evt.content || '', evt.session_id)
        else if (evt.type === 'error') handlers.onError?.(evt.message || '系统出问题了，请稍后重试')
      } catch {
        /* 忽略单个事件解析失败 */
      }
    }
  }
}
