import { mount } from '@vue/test-utils'
import { createPinia } from 'pinia'
import ElementPlus from 'element-plus'
import { beforeEach, describe, expect, it, vi } from 'vitest'

vi.mock('vue-router', () => ({
  useRouter: () => ({ push: vi.fn() }),
}))

vi.mock('element-plus', async (importOriginal) => {
  const actual = await importOriginal<typeof import('element-plus')>()
  return { ...actual, ElMessage: { warning: vi.fn(), success: vi.fn(), error: vi.fn() } }
})

import { ElMessage } from 'element-plus'
import RegisterView from '@/views/RegisterView.vue'

describe('RegisterView', () => {
  beforeEach(() => vi.clearAllMocks())

  it('两次密码不一致 → 提示警告', async () => {
    const wrapper = mount(RegisterView, {
      global: { plugins: [createPinia(), ElementPlus], stubs: { 'router-link': true } },
    })
    // 填用户名密码但确认密码不同
    const inputs = wrapper.findAll('input')
    await inputs[0].setValue('testuser')
    await inputs[1].setValue('pass123')
    await inputs[2].setValue('different')
    await wrapper.find('button.submit').trigger('click')
    expect(ElMessage.warning).toHaveBeenCalled()
  })

  it('密码过短 → 提示警告', async () => {
    const wrapper = mount(RegisterView, {
      global: { plugins: [createPinia(), ElementPlus], stubs: { 'router-link': true } },
    })
    const inputs = wrapper.findAll('input')
    await inputs[0].setValue('testuser')
    await inputs[1].setValue('123')
    await inputs[2].setValue('123')
    await wrapper.find('button.submit').trigger('click')
    expect(ElMessage.warning).toHaveBeenCalled()
  })
})
