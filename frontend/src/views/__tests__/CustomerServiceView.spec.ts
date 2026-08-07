import { shallowMount, flushPromises } from '@vue/test-utils'
import { defineComponent } from 'vue'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import CustomerServiceView from '@/views/CustomerServiceView.vue'

// ---- mock composables ----
const sessionMock = vi.hoisted(() => {
  const currentSessionId = { value: '' }
  const restoreSession = vi.fn(() => '')
  const setSession = vi.fn((sid: string) => { currentSessionId.value = sid })
  const clearSession = vi.fn(() => { currentSessionId.value = '' })
  const isBlankNewSession = vi.fn(() => false)
  const loadSessions = vi.fn()
  const loadMessages = vi.fn()
  const removeSession = vi.fn()
  const ensureSession = vi.fn()
  return {
    currentSessionId, restoreSession, setSession, clearSession, isBlankNewSession,
    loadSessions, loadMessages, removeSession, ensureSession,
  }
})

const chatMock = vi.hoisted(() => ({
  messages: { value: [] as { id: string; role: string; content: string }[] },
  sending: { value: false },
  progress: { value: '' },
  confirmVisible: { value: false },
  send: vi.fn(),
  sendConfirm: vi.fn(),
  clear: vi.fn(),
  loadHistory: vi.fn(),
}))

vi.mock('@/composables/useSession', () => ({
  useSession: () => sessionMock,
}))

vi.mock('@/composables/useChat', () => ({
  useChat: () => chatMock,
}))

vi.mock('@/composables/useAuth', () => ({
  useAuth: () => ({
    user: { value: { username: 'user_1' } },
    isAdmin: { value: false },
    logout: vi.fn(),
  }),
}))

vi.mock('vue-router', () => ({
  useRouter: () => ({ push: vi.fn() }),
}))

// el-popconfirm stub：渲染 reference slot，测试中通过 $emit('confirm') 触发删除
const ElPopconfirmStub = defineComponent({
  name: 'ElPopconfirm',
  template: '<div class="popconfirm-stub"><slot name="reference" /></div>',
})

const ITEMS = [
  { session_id: 's1', title: '会话一', updated_at: '2026-08-07T12:00:00', intent: null },
  { session_id: 's2', title: '会话二', updated_at: '2026-08-07T13:00:00', intent: null },
]

function mountView() {
  return shallowMount(CustomerServiceView, {
    global: {
      stubs: {
        ChatPanel: true,
        ChatInput: true,
        ElPopconfirm: ElPopconfirmStub,
        // 布局组件 stub 必须渲染 slot，否则内部 .session-item / 列表内容全部丢失
        'el-container': { template: '<div><slot /></div>' },
        'el-aside': { template: '<div><slot /></div>' },
        'el-main': { template: '<div><slot /></div>' },
        'el-button': { template: '<button @click="$emit(\'click\', $event)"><slot /></button>' },
        'el-tag': { template: '<span><slot /></span>' },
      },
    },
  })
}

describe('CustomerServiceView 会话编排', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    sessionMock.currentSessionId.value = ''
    sessionMock.restoreSession.mockReturnValue('')
    sessionMock.isBlankNewSession.mockReturnValue(false)
    chatMock.messages.value = []
  })

  it('挂载时加载列表并恢复 localStorage 指向的有效会话', async () => {
    sessionMock.restoreSession.mockReturnValue('s1')
    sessionMock.currentSessionId.value = 's1'
    sessionMock.loadSessions.mockResolvedValue({ items: ITEMS, total: 2 })

    const wrapper = mountView()
    await flushPromises()

    expect(sessionMock.loadSessions).toHaveBeenCalled()
    expect(chatMock.loadHistory).toHaveBeenCalledWith('s1')
    expect(wrapper.text()).toContain('会话一')
    expect(wrapper.text()).toContain('会话二')
  })

  it('首次进入（无任何记录）恢复最近会话第一条', async () => {
    sessionMock.loadSessions.mockResolvedValue({ items: ITEMS, total: 2 })

    mountView()
    await flushPromises()

    expect(sessionMock.setSession).toHaveBeenCalledWith('s1')
    expect(chatMock.loadHistory).toHaveBeenCalledWith('s1')
  })

  it('主动新建会话后刷新保持空白，不覆盖为新会话', async () => {
    sessionMock.isBlankNewSession.mockReturnValue(true)
    sessionMock.loadSessions.mockResolvedValue({ items: ITEMS, total: 2 })

    mountView()
    await flushPromises()

    expect(sessionMock.setSession).not.toHaveBeenCalled()
    expect(chatMock.loadHistory).not.toHaveBeenCalled()
  })

  it('localStorage 指向的会话已被删除/过期时回退列表第一条', async () => {
    sessionMock.restoreSession.mockReturnValue('ghost')
    sessionMock.currentSessionId.value = 'ghost' // 不在列表
    sessionMock.loadSessions.mockResolvedValue({ items: ITEMS, total: 2 })

    mountView()
    await flushPromises()

    expect(sessionMock.setSession).toHaveBeenCalledWith('s1')
    expect(chatMock.loadHistory).toHaveBeenCalledWith('s1')
  })

  it('点击历史会话切换并加载对应历史', async () => {
    sessionMock.restoreSession.mockReturnValue('s1')
    sessionMock.currentSessionId.value = 's1'
    sessionMock.loadSessions.mockResolvedValue({ items: ITEMS, total: 2 })

    const wrapper = mountView()
    await flushPromises()
    chatMock.loadHistory.mockClear()

    const items = wrapper.findAll('.session-item')
    await items[1].trigger('click')

    expect(sessionMock.setSession).toHaveBeenCalledWith('s2')
    expect(chatMock.loadHistory).toHaveBeenCalledWith('s2')
  })

  it('点击同一会话不重复加载', async () => {
    sessionMock.restoreSession.mockReturnValue('s1')
    sessionMock.currentSessionId.value = 's1'
    sessionMock.loadSessions.mockResolvedValue({ items: ITEMS, total: 2 })

    const wrapper = mountView()
    await flushPromises()
    chatMock.loadHistory.mockClear()

    await wrapper.findAll('.session-item')[0].trigger('click')

    expect(sessionMock.setSession).not.toHaveBeenCalled()
    expect(chatMock.loadHistory).not.toHaveBeenCalled()
  })

  it('新建会话清空当前对话与 session 状态', async () => {
    sessionMock.loadSessions.mockResolvedValue({ items: [], total: 0 })

    const wrapper = mountView()
    await flushPromises()

    await wrapper.find('.new-session').trigger('click')

    expect(chatMock.clear).toHaveBeenCalled()
    expect(sessionMock.clearSession).toHaveBeenCalled()
  })

  it('删除会话后刷新列表；删除当前会话则回退第一条', async () => {
    sessionMock.restoreSession.mockReturnValue('s1')
    sessionMock.currentSessionId.value = 's1'
    sessionMock.removeSession.mockResolvedValue({ msg: '已删除' })
    sessionMock.loadSessions
      .mockResolvedValueOnce({ items: ITEMS, total: 2 })   // 挂载列表
      .mockResolvedValueOnce({ items: [ITEMS[1]], total: 1 }) // 删除后列表

    const wrapper = mountView()
    await flushPromises()

    const popconfirms = wrapper.findAllComponents(ElPopconfirmStub)
    expect(popconfirms).toHaveLength(2)
    await popconfirms[0].vm.$emit('confirm')

    await flushPromises()

    expect(sessionMock.removeSession).toHaveBeenCalledWith('s1')
    expect(chatMock.clear).toHaveBeenCalled()
    expect(sessionMock.setSession).toHaveBeenLastCalledWith('s2')
    expect(chatMock.loadHistory).toHaveBeenLastCalledWith('s2')
  })

  it('删除非当前会话不影响当前会话', async () => {
    sessionMock.restoreSession.mockReturnValue('s1')
    sessionMock.currentSessionId.value = 's1'
    sessionMock.removeSession.mockResolvedValue({ msg: '已删除' })
    sessionMock.loadSessions.mockResolvedValue({ items: ITEMS, total: 2 })

    const wrapper = mountView()
    await flushPromises()

    const popconfirms = wrapper.findAllComponents(ElPopconfirmStub)
    await popconfirms[1].vm.$emit('confirm')
    await flushPromises()

    expect(sessionMock.removeSession).toHaveBeenCalledWith('s2')
    expect(chatMock.clear).not.toHaveBeenCalled()
  })
})
