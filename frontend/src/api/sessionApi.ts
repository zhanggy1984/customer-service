/** 会话 / 消息接口 */
import client from './client'

export interface CreateSessionResp {
  session_id: string
}

export interface SendMessageResp {
  session_id: string
  reply: string
}

export const createSession = () =>
  client.post('/sessions') as Promise<CreateSessionResp>

export const sendMessage = (sid: string, content: string) =>
  client.post(`/sessions/${sid}/messages`, { content }) as Promise<SendMessageResp>
