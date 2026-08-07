import { describe, expect, it } from 'vitest'

import { formatTime } from '@/utils/formatTime'

/** 构造距今天 offsetDays 天、指定时分（本地时区）的 ISO 字符串 */
function isoFor(offsetDays: number, hour = 12, min = 0): string {
  const d = new Date()
  d.setDate(d.getDate() + offsetDays)
  d.setHours(hour, min, 0, 0)
  return d.toISOString()
}

describe('formatTime', () => {
  it('空值返回空串', () => {
    expect(formatTime()).toBe('')
    expect(formatTime(null)).toBe('')
    expect(formatTime('')).toBe('')
  })

  it('非法日期返回空串', () => {
    expect(formatTime('not-a-date')).toBe('')
  })

  it('今天显示 HH:mm', () => {
    expect(formatTime(isoFor(0, 10, 5))).toBe('10:05')
    expect(formatTime(isoFor(0, 23, 59))).toBe('23:59')
  })

  it('昨天显示"昨天 HH:mm"', () => {
    expect(formatTime(isoFor(-1, 8, 30))).toBe('昨天 08:30')
  })

  it('更早日期显示 M-d', () => {
    const d = new Date()
    d.setDate(d.getDate() - 3)
    const expectVal = `${d.getMonth() + 1}-${d.getDate()}`
    expect(formatTime(isoFor(-3, 9, 0))).toBe(expectVal)
  })
})
