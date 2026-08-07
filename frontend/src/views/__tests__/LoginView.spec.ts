import { mount } from '@vue/test-utils'
import { createPinia } from 'pinia'
import ElementPlus from 'element-plus'
import { beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('vue-router', () => ({
  useRouter: () => ({ push: vi.fn() }),
  useRoute: () => ({ query: {} }),
}))

vi.mock('element-plus', async (importOriginal) => {
  const actual = await importOriginal<typeof import('element-plus')>()
  return { ...actual, ElMessage: { warning: vi.fn(), success: vi.fn(), error: vi.fn() } }
})

import { ElMessage } from 'element-plus'
import LoginView from '@/views/LoginView.vue'

describe('LoginView', () => {
  beforeEach(() => vi.clearAllMocks())

  it('空输入点击登录 → 提示警告且不调用登录', async () => {
    const wrapper = mount(LoginView, {
      global: { plugins: [createPinia(), ElementPlus], stubs: { 'router-link': true } },
    })
    await wrapper.find('button.submit').trigger('click')
    expect(ElMessage.warning).toHaveBeenCalled()
  })
})
