/**
 * 业务枚举 → 中文显示映射（集中定义，展示层复用）。
 *
 * 注意：PAID/SHIPPED 等英文枚举是系统契约（DB ENUM / 对接层 / Agent 均依赖），
 * 此处仅做展示层翻译，不改数据层。
 */

/** 订单状态 */
export const ORDER_STATUS_MAP: Record<string, string> = {
  PAID: '已付款',
  SHIPPED: '已发货',
  DELIVERED: '已签收',
  CANCELLED: '已取消',
}

/** 商品明细状态（支持部分退货） */
export const ORDER_ITEM_STATUS_MAP: Record<string, string> = {
  NORMAL: '正常',
  RETURN_REQUESTED: '退货申请中',
  RETURNED: '已退货',
  REFUNDED: '已退款',
}

/** 下拉选项：值保持英文枚举，label 显示中文 */
export const ORDER_STATUS_OPTIONS = Object.entries(ORDER_STATUS_MAP).map(([value, label]) => ({ value, label }))

/** 兜底取原值 */
export function statusText(map: Record<string, string>, value: string): string {
  return map[value] || value
}
