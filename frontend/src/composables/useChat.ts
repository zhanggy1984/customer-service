/** 聊天状态：SSE 流式 + 进度 + 确认按钮 + 逐字打字效果 */
import { ref } from 'vue'

import { useSession } from './useSession'
import { streamChat } from './useSSE'

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  streaming?: boolean
}

let idSeq = 0
const nextId = () => `m_${Date.now()}_${idSeq++}`

export function useChat() {
  const messages = ref<ChatMessage[]>([])
  const sending = ref(false)
  const progress = ref('')
  const confirmVisible = ref(false)
  const { ensureSession, setSession } = useSession()

  function setContent(id: string, content: string) {
    const m = messages.value.find((x) => x.id === id)
    if (m) {
      m.content = content
      m.streaming = false
    }
  }

  async function typeWriter(id: string, full: string) {
    const m = messages.value.find((x) => x.id === id)
    if (!m) return
    m.streaming = true
    let i = 0
    while (i < full.length && m.streaming) {
      m.content = full.slice(0, i)
      i += 2
      await new Promise((r) => setTimeout(r, 16))
    }
    m.content = full
    m.streaming = false
  }

  async function send(content: string) {
    const text = content.trim()
    if (!text || sending.value) return

    sending.value = true
    progress.value = ''
    confirmVisible.value = false
    messages.value.push({ id: nextId(), role: 'user', content: text })
    const assistantId = nextId()
    messages.value.push({ id: assistantId, role: 'assistant', content: '', streaming: true })

    try {
      const sid = await ensureSession()
      await streamChat(sid, text, {
        onStatus: (msg) => {
          progress.value = msg
        },
        onAction: (action) => {
          if (action === 'confirm') confirmVisible.value = true
        },
        onDone: (reply, newSid) => {
          if (newSid) setSession(newSid)
          void typeWriter(assistantId, reply || '（空回复）')
        },
        onError: (errMsg) => setContent(assistantId, errMsg),
      })
    } catch {
      setContent(assistantId, '发送失败，请稍后重试。')
    } finally {
      sending.value = false
      progress.value = ''
      // 不在此清除确认按钮：由 action 事件控制出现，
      // 用户点击确认/取消或下一条消息时在 send 开头清除
    }
  }

  async function sendConfirm() {
    await send('确认')
  }

  function clear() {
    messages.value = []
    confirmVisible.value = false
  }

  return { messages, sending, progress, confirmVisible, send, sendConfirm, clear }
}
