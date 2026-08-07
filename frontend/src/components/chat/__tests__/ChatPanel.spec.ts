import { mount } from '@vue/test-utils'
import { describe, expect, it } from 'vitest'

import ChatPanel from '@/components/chat/ChatPanel.vue'

describe('ChatPanel', () => {
  it('渲染用户和助手气泡', () => {
    const wrapper = mount(ChatPanel, {
      props: {
        messages: [
          { id: '1', role: 'user', content: '你好' },
          { id: '2', role: 'assistant', content: '您好，请问有什么可以帮您？' },
        ],
        sending: false,
        progress: '',
      },
    })
    const bubbles = wrapper.findAll('.bubble')
    expect(bubbles).toHaveLength(2)
    expect(bubbles[0].text()).toContain('你好')
    expect(bubbles[1].text()).toContain('您好')
    expect(wrapper.findAll('.msg-row.user')).toHaveLength(1)
    expect(wrapper.findAll('.msg-row.assistant')).toHaveLength(1)
  })

  it('progress 显示进度条', () => {
    const wrapper = mount(ChatPanel, {
      props: { messages: [], sending: true, progress: '正在查询订单...' },
    })
    expect(wrapper.text()).toContain('正在查询订单...')
  })

  it('空消息显示欢迎语', () => {
    const wrapper = mount(ChatPanel, {
      props: { messages: [], sending: false, progress: '' },
    })
    expect(wrapper.text()).toContain('智能客服')
  })
})
