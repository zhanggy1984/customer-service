import { mount } from '@vue/test-utils'
import ElementPlus from 'element-plus'
import { describe, expect, it } from 'vitest'

import ChatInput from '@/components/chat/ChatInput.vue'

describe('ChatInput', () => {
  it('点击发送 emit send 且清空输入', async () => {
    const wrapper = mount(ChatInput, { props: { disabled: false }, global: { plugins: [ElementPlus] } })
    const textarea = wrapper.find('textarea')
    await textarea.setValue('你好')
    await wrapper.find('.send-btn').trigger('click')
    const emitted = wrapper.emitted('send')
    expect(emitted).toBeTruthy()
    expect(emitted![0]).toEqual(['你好'])
  })

  it('空内容不发送', async () => {
    const wrapper = mount(ChatInput, { props: { disabled: false }, global: { plugins: [ElementPlus] } })
    await wrapper.find('.send-btn').trigger('click')
    expect(wrapper.emitted('send')).toBeFalsy()
  })

  it('disabled 时不发送', async () => {
    const wrapper = mount(ChatInput, { props: { disabled: true }, global: { plugins: [ElementPlus] } })
    const textarea = wrapper.find('textarea')
    await textarea.setValue('测试')
    await wrapper.find('.send-btn').trigger('click')
    expect(wrapper.emitted('send')).toBeFalsy()
  })
})
