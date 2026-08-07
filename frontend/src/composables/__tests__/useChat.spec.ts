import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'

import * as sessionApi from '@/api/sessionApi'
import { useAuthStore } from '@/stores/authStore'
import { useChat } from '@/composables/useChat'

vi.mock('@/api/sessionApi', () => ({
  createSession: vi.fn(),
  listSessions: vi.fn(),
  getSessionMessages: vi.fn(),
  deleteSession: vi.fn(),
}))

describe('useChat.loadHistory', () => {
  beforeEach(() => {
    localStorage.clear()
    setActivePinia(createPinia())
    const auth = useAuthStore()
    auth.user = { id: 2, username: 'user_1', role: 'user', phone: '13800000001' }
  })

  it('拉取历史消息并映射为 ChatMessage 替换列表', async () => {
    vi.mocked(sessionApi.getSessionMessages).mockResolvedValue({
      session_id: 's1',
      intent: null,
      messages: [
        { role: 'user', content: '我要退货', ts: '2026-08-07T12:00:00Z' },
        { role: 'assistant', content: '请问退货原因？', ts: '2026-08-07T12:00:01Z' },
      ],
    })
    const { loadHistory, messages } = useChat()

    await loadHistory('s1')

    expect(messages.value).toHaveLength(2)
    expect(messages.value[0]).toMatchObject({ role: 'user', content: '我要退货' })
    expect(messages.value[1]).toMatchObject({ role: 'assistant', content: '请问退货原因？' })
    expect(messages.value[0].id).toContain('h_s1_') // 历史消息 id 带会话前缀
  })

  it('空消息列表替换为空数组', async () => {
    vi.mocked(sessionApi.getSessionMessages).mockResolvedValue({
      session_id: 's1',
      intent: null,
      messages: [],
    })
    const { loadHistory, messages } = useChat()

    await loadHistory('s1')

    expect(messages.value).toHaveLength(0)
  })
})
