/** 会话管理：懒创建 + 切换 + 关闭；session_id 按用户持久化（刷新 / 重新登录可恢复） */
import { ref } from 'vue'

import { createSession, deleteSession, getSessionMessages, listSessions } from '@/api/sessionApi'
import { useAuthStore } from '@/stores/authStore'

/** localStorage key 按用户隔离，避免 A 的会话串到 B */
function sessionStorageKey(): string {
  const auth = useAuthStore()
  return `cs_session_${auth.user?.id ?? 'anon'}`
}

const currentSessionId = ref<string>('')

export function useSession() {
  /** 从 localStorage 按当前用户恢复上次会话；组件挂载 / 用户切换后调用 */
  function restoreSession(): string {
    const sid = localStorage.getItem(sessionStorageKey()) || ''
    currentSessionId.value = sid
    return sid
  }

  async function ensureSession(): Promise<string> {
    if (!currentSessionId.value) restoreSession() // 兜底：localStorage 已有则复用，避免误建新会话
    if (!currentSessionId.value) {
      const data = await createSession()
      currentSessionId.value = data.session_id
      localStorage.setItem(sessionStorageKey(), currentSessionId.value)
    }
    return currentSessionId.value
  }

  function setSession(sid: string) {
    currentSessionId.value = sid
    localStorage.setItem(sessionStorageKey(), sid)
  }

  function clearSession() {
    currentSessionId.value = ''
    // 置空而非删除 key：区分"用户主动新建会话"（key 存在但空）与"从未有过会话"（key 不存在），
    // 刷新后 restoreActiveSession 据此决定保持空白还是恢复最近会话
    localStorage.setItem(sessionStorageKey(), '')
  }

  /** 是否"主动新建会话后刷新"：当前无会话且 key 存在（值为空串标记） */
  function isBlankNewSession(): boolean {
    return !currentSessionId.value && localStorage.getItem(sessionStorageKey()) !== null
  }

  function loadSessions() {
    return listSessions()
  }

  function loadMessages(sid: string) {
    return getSessionMessages(sid)
  }

  function removeSession(sid: string) {
    return deleteSession(sid)
  }

  return {
    currentSessionId,
    restoreSession,
    ensureSession,
    setSession,
    clearSession,
    isBlankNewSession,
    loadSessions,
    loadMessages,
    removeSession,
  }
}
