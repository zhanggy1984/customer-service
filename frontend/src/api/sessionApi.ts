/** 会话 / 消息接口 */
import client from './client'

export interface CreateSessionResp {
  session_id: string
}

export interface SendMessageResp {
  session_id: string
  reply: string
}

export interface SessionItem {
  session_id: string
  title: string
  updated_at: string | null // 后端语义：最后保存/活跃时间（conversation_history 每次 DELETE+INSERT 刷新）
  intent?: string | null
}

export interface SessionListResp {
  items: SessionItem[]
  total: number
}

export interface SessionMessage {
  role: 'user' | 'assistant'
  content: string
  ts: string
}

export interface SessionMessagesResp {
  session_id: string
  intent?: string | null
  messages: SessionMessage[]
}

export const createSession = () =>
  client.post('/sessions') as Promise<CreateSessionResp>

export const listSessions = () =>
  client.get('/sessions') as Promise<SessionListResp>

export const getSessionMessages = (sid: string) =>
  client.get(`/sessions/${sid}/messages`) as Promise<SessionMessagesResp>

export const sendMessage = (sid: string, content: string) =>
  client.post(`/sessions/${sid}/messages`, { content }) as Promise<SendMessageResp>

export const deleteSession = (sid: string) =>
  client.delete(`/sessions/${sid}`) as Promise<{ msg: string }>
