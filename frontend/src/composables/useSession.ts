/** 会话管理：懒创建 + 切换 + 关闭（Phase 1 单会话，后续接入多会话列表） */
import { ref } from 'vue'

import { createSession } from '@/api/sessionApi'

const currentSessionId = ref<string>('')

export function useSession() {
  async function ensureSession(): Promise<string> {
    if (!currentSessionId.value) {
      const data = await createSession()
      currentSessionId.value = data.session_id
    }
    return currentSessionId.value
  }

  function setSession(sid: string) {
    currentSessionId.value = sid
  }

  function clearSession() {
    currentSessionId.value = ''
  }

  return { currentSessionId, ensureSession, setSession, clearSession }
}
