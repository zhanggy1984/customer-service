/** Admin 接口：知识库 + 订单管理 */
import client from './client'

export interface KnowledgeDoc {
  source: string
  content: string
  chunk_count: number
  sync_status: 'ok' | 'pending'
  updated_by: string | null
  created_at: string | null
  updated_at: string | null
}

export interface OrderItem {
  id: number
  item_id: string
  name: string
  price: number
  quantity: number
  returnable: number
  status: string
}

export interface Order {
  id: number
  order_id: string
  user_id: number
  status: string
  total_amount: string
  shipping_address: string
  created_at: string
  items: OrderItem[]
}

export interface OrderItemIn {
  item_id: string
  name: string
  price: number
  quantity: number
  returnable: boolean
}

export const listKnowledge = () =>
  client.get('/admin/knowledge') as Promise<{ items: KnowledgeDoc[]; total: number }>

export const uploadKnowledge = (title: string, content: string) =>
  client.post('/admin/knowledge', { title, content }) as Promise<{ count: number }>

export const updateKnowledge = (source: string, content: string) =>
  client.put(`/admin/knowledge/${encodeURIComponent(source)}`, { content }) as Promise<{ msg: string; chunks: number }>

export const deleteKnowledge = (source: string) =>
  client.delete(`/admin/knowledge/${encodeURIComponent(source)}`) as Promise<{ msg: string }>

/** 全量对账：从 MySQL 重建全部文档 chunks + 清理孤儿（异常恢复时手动触发） */
export const syncKnowledge = () =>
  client.post('/admin/knowledge/sync') as Promise<{ synced: number; orphan_removed: number }>

export const listOrders = () =>
  client.get('/admin/orders') as Promise<{ items: Order[] }>

export const createOrder = (data: {
  order_id: string
  user_id: number
  status: string
  total_amount: number
  shipping_address: string
  items: OrderItemIn[]
}) => client.post('/admin/orders', data) as Promise<{ msg: string; order_id: string }>

export const updateOrder = (orderId: string, data: { status?: string }) =>
  client.put(`/admin/orders/${orderId}`, data) as Promise<{ msg: string }>

export const deleteOrder = (orderId: string) =>
  client.delete(`/admin/orders/${orderId}`) as Promise<{ msg: string }>

/** 测试数据一键重置：清空退货/退款/工单，恢复种子订单（测试前调用） */
export const resetDemoData = () =>
  client.post('/admin/reset-demo') as Promise<{ msg: string; orders: number }>
