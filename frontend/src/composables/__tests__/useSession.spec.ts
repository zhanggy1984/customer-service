import { beforeEach, describe, expect, it, vi } from 'vitest'
import { createPinia, setActivePinia } from 'pinia'

import * as sessionApi from '@/api/sessionApi'
import { useAuthStore } from '@/stores/authStore'
import { useSession } from '@/composables/useSession'

vi.mock('@/api/sessionApi', () => ({
  createSession: vi.fn(),
  listSessions: vi.fn(),
  getSessionMessages: vi.fn(),
  deleteSession: vi.fn(),
}))

const USER_1 = { id: 2, username: 'user_1', role: 'user' as const, phone: '13800000001' }
const USER_2 = { id: 3, username: 'user_2', role: 'user' as const, phone: '13800000002' }

describe('useSession', () => {
  beforeEach(() => {
    localStorage.clear()
    vi.clearAllMocks() // 避免 createSession 等 mock 调用计数跨测试累计
    // 通过 cs_user 让 authStore 初始化 user（与真实登录流程一致，避免手动赋值失效）
    localStorage.setItem('cs_user', JSON.stringify(USER_1))
    setActivePinia(createPinia())
    // 重置 module 级 currentSessionId
    const { clearSession } = useSession()
    clearSession()
  })

  it('ensureSession 创建会话并持久化到当前用户 key', async () => {
    vi.mocked(sessionApi.createSession).mockResolvedValue({ session_id: 's1' })
    const { ensureSession } = useSession()

    const sid = await ensureSession()

    expect(sid).toBe('s1')
    expect(localStorage.getItem('cs_session_2')).toBe('s1')
  })

  it('ensureSession 已存在会话时不重复创建', async () => {
    localStorage.setItem('cs_session_2', 'existing')
    const { ensureSession } = useSession()

    const sid = await ensureSession()

    expect(sid).toBe('existing')
    expect(sessionApi.createSession).not.toHaveBeenCalled()
  })

  it('setSession 按用户隔离写入 localStorage', () => {
    const { setSession } = useSession()

    setSession('sA')

    expect(localStorage.getItem('cs_session_2')).toBe('sA')
    expect(localStorage.getItem('cs_session_3')).toBeNull() // 其他用户 key 不受影响
  })

  it('restoreSession 从 localStorage 恢复当前用户会话', () => {
    localStorage.setItem('cs_session_2', 'restored')
    const { restoreSession } = useSession()

    expect(restoreSession()).toBe('restored')
  })

  it('clearSession 置空而非删除 key（保留"主动新建会话"标记）', () => {
    localStorage.setItem('cs_session_2', 'x')
    const { clearSession } = useSession()

    clearSession()

    expect(localStorage.getItem('cs_session_2')).toBe('')
  })

  it('isBlankNewSession 区分"主动新建"与"从未有过"', () => {
    const { clearSession, isBlankNewSession } = useSession()
    // key 存在但空（主动新建后）→ true
    clearSession()
    expect(isBlankNewSession()).toBe(true)
    // key 不存在（首次进入）→ false
    localStorage.removeItem('cs_session_2')
    expect(isBlankNewSession()).toBe(false)
  })

  it('切换用户后 key 按新用户隔离', () => {
    const { setSession } = useSession()
    setSession('u2')
    expect(localStorage.getItem('cs_session_2')).toBe('u2')

    useAuthStore().user = USER_2
    const { setSession: setSessionOf3 } = useSession()
    setSessionOf3('u3')

    expect(localStorage.getItem('cs_session_3')).toBe('u3')
    expect(localStorage.getItem('cs_session_2')).toBe('u2') // 旧用户数据不被覆盖
  })
})
