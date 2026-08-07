/**
 * UI 端到端验证脚本（Playwright 驱动系统 Chrome）。
 * 验证：登录 → 聊天 → 退货全流程(进度条/确认按钮/退单号) → 意图切换。
 * 运行: cd frontend && node ui-verify.mjs
 */
import { chromium } from 'playwright-core'

const CHROME = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe'
// 入口默认 dev server(5173)，发布验证用 UI_BASE=http://localhost 走 nginx
const BASE = process.env.UI_BASE || 'http://localhost:5173'
const TEXTAREA = 'textarea[placeholder*="请输入您的问题"]'

async function main() {
  const browser = await chromium.launch({ executablePath: CHROME, headless: true })
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 } })
  const results = []
  const step = (name, ok, extra = '') => {
    results.push(ok)
    console.log(`${ok ? '✅' : '❌'} ${name}${extra ? ` — ${extra}` : ''}`)
  }
  const waitText = (text, timeout = 60000) =>
    page.waitForFunction((t) => document.body.innerText.includes(t), text, { timeout })

  try {
    // 1. 登录页
    await page.goto(`${BASE}/login`)
    await page.waitForSelector('text=智能客服', { timeout: 15000 })
    step('登录页加载', true)

    // 2. 登录 user_1
    await page.fill('input[placeholder="请输入用户名"]', 'user_1')
    await page.fill('input[placeholder="请输入密码"]', '123456')
    await page.click('button.submit')
    await page.waitForURL('**/chat', { timeout: 15000 })
    step('登录并跳转聊天页', true)

    // 3. 退货流程第 1 轮：问原因
    await page.fill(TEXTAREA, '我要退货 ORD-20240801-001')
    await page.press(TEXTAREA, 'Enter')
    await waitText('退货原因')
    step('退货流程: 追问原因', true)

    // 4. 第 2 轮：确认信息 + 确认按钮
    await page.fill(TEXTAREA, '质量问题')
    await page.press(TEXTAREA, 'Enter')
    await page.waitForSelector('.confirm-bar', { timeout: 60000 })
    step('确认按钮(ConfirmButton)出现', true)

    // 5. 点确认 → 退单号
    await page.click('text=确认提交')
    await waitText('RC-')
    step('点确认后收到退单号 RC-', true)

    // 6. 意图切换：退货中途查订单
    await page.fill(TEXTAREA, '查一下订单 ORD-20240805-002')
    await page.press(TEXTAREA, 'Enter')
    await waitText('已发货')
    step('意图切换: 查订单成功', true)

    // 6b. 订单状态显示中文（后端 STATUS_DESC 中文，无英文枚举）
    const statusOk = await page.evaluate(() => {
      const t = document.body.innerText
      return /已付款|已发货|已签收|已取消/.test(t) && !/PAID|SHIPPED|DELIVERED|CANCELLED/.test(t)
    })
    step('订单状态显示中文(聊天)', statusOk)

    // 7. 刷新保持登录
    await page.reload()
    await page.waitForSelector('textarea', { timeout: 15000 })
    step('刷新后登录态保持', true)

    // 8. 登出
    await page.click('text=退出')
    await page.waitForURL('**/login', { timeout: 15000 })
    step('登出跳回登录页', true)

    await browser.close()
    const pass = results.filter(Boolean).length
    console.log(`\n===== UI 验证: ${pass}/${results.length} 通过 =====`)
    process.exit(pass === results.length ? 0 : 1)
  } catch (e) {
    await page.screenshot({ path: 'ui-verify-fail.png' })
    console.error('❌ 脚本异常:', e.message)
    await browser.close()
    process.exit(1)
  }
}

main()
