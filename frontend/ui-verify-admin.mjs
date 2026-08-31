/**
 * Admin 管理后台 UI 验证：登录 → 知识库文档管理（列表/同步按钮/编辑弹窗）→ 订单管理（中文状态）。
 * 运行: cd frontend && node ui-verify-admin.mjs
 */
import { chromium } from 'playwright-core'

const CHROME = 'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe'
// 入口默认 dev server(5173)，发布验证用 UI_BASE=http://localhost 走 nginx
const BASE = process.env.UI_BASE || 'http://localhost:5173'

async function main() {
  const browser = await chromium.launch({ executablePath: CHROME, headless: true })
  const page = await browser.newPage({ viewport: { width: 1280, height: 800 } })
  const results = []
  const step = (n, ok, e = '') => {
    results.push(ok)
    console.log(`${ok ? '✅' : '❌'} ${n}${e ? ` — ${e}` : ''}`)
  }

  try {
    await page.goto(`${BASE}/login`)
    await page.waitForSelector('text=智能客服', { timeout: 15000 })
    await page.fill('input[placeholder="请输入用户名"]', 'admin')
    await page.fill('input[placeholder="请输入密码"]', '123456')
    await page.click('button.submit')
    await page.waitForURL('**/chat', { timeout: 15000 })
    step('admin 登录', true)

    await page.click('text=管理后台')
    await page.waitForURL('**/admin', { timeout: 15000 })
    step('进入管理后台', true)

    // ===== 知识库文档管理 =====
    await page.waitForFunction(() => document.querySelectorAll('.el-table__body tr').length > 0, { timeout: 20000 })
    step('知识库文档列表加载(≥1 篇)', true)

    const hasSyncBtn = await page.$('text=同步知识库')
    step('同步知识库按钮存在', !!hasSyncBtn)

    // 文档列表含"编辑"操作 + 同步状态标签
    const kbState = await page.evaluate(() => {
      const t = document.querySelector('.el-table')?.textContent || ''
      return t.includes('编辑') && /已同步|待同步/.test(t)
    })
    step('文档列表含编辑/同步状态标签', kbState)

    // 编辑弹窗
    await page.click('.el-table .el-button:has-text("编辑")')
    await page.waitForSelector('.el-dialog', { timeout: 10000 })
    const dlgTitle = await page.textContent('.el-dialog__title')
    step(`编辑弹窗打开(${dlgTitle})`, dlgTitle.includes('编辑文档'))
    await page.click('.el-dialog .el-button:has-text("取消")')

    // ===== 订单管理 =====
    await page.click('text=订单管理')
    await page.waitForFunction(() => document.querySelectorAll('.el-table__body tr').length >= 5, { timeout: 20000 })
    step('订单列表加载(≥5 条种子订单)', true)

    // 断言订单状态显示为中文（仅检查订单表格，避免知识库文档里的英文枚举误判）
    const statusCheck = await page.evaluate(() => {
      const tables = Array.from(document.querySelectorAll('.el-table')).map((t) => t.textContent || '')
      const orderTbl = tables.find((t) => t.includes('订单号')) || ''
      return {
        hasChinese: /已付款|已发货|已签收|已取消/.test(orderTbl),
        hasEnglish: /PAID|SHIPPED|DELIVERED|CANCELLED/.test(orderTbl),
      }
    })
    step('订单状态显示中文(订单表格)', statusCheck.hasChinese && !statusCheck.hasEnglish)

    // 新建订单弹窗（布局：表头齐全 + 表单项不挤）
    await page.click('button:has-text("新建订单")')
    await page.waitForSelector('.el-dialog:has-text("新建订单")', { timeout: 10000 })
    const orderDlg = await page.textContent('.el-dialog:has-text("新建订单")')
    const dlgOk =
      orderDlg.includes('商品明细') &&
      orderDlg.includes('SKU 编号') &&
      orderDlg.includes('商品名称') &&
      orderDlg.includes('单价') &&
      orderDlg.includes('数量') &&
      orderDlg.includes('可退')
    step('新建订单弹窗布局(商品明细表头齐全)', dlgOk)

    // 对齐断言：表头各列左边界与数据行对应列左边界一致（el-table 天然对齐）
    const alignOk = await page.evaluate(() => {
      const ths = Array.from(document.querySelectorAll('.item-table thead th'))
      const tds = Array.from(document.querySelectorAll('.item-table tbody td'))
      if (ths.length === 0 || tds.length === 0) return false
      const xs = (els) => els.map((el) => Math.round(el.getBoundingClientRect().x))
      const thX = xs(ths)
      const tdX = xs(tds.slice(0, ths.length))
      return thX.every((x, i) => Math.abs(x - tdX[i]) <= 2)
    })
    step('商品明细表头与数据列对齐', alignOk)
    await page.click('.el-dialog:has-text("新建订单") .el-button:has-text("取消")')
    await page.waitForSelector('.el-dialog:has-text("新建订单")', { state: 'hidden', timeout: 10000 }).catch(() => {})

    // 一键重置测试数据（二次确认）
    await page.click('text=重置测试数据')
    await page.waitForSelector('.el-message-box', { timeout: 10000 })
    const confirmMsg = await page.textContent('.el-message-box__message')
    step('重置弹窗二次确认', confirmMsg.includes('清空全部退货单'))
    await page.click('.el-message-box .el-button--primary')
    await page.waitForFunction(() => document.body.innerText.includes('已重置'), { timeout: 15000 })
    step('重置成功提示(已重置)', true)

    // 返回聊天
    await page.click('text=返回聊天')
    await page.waitForURL('**/chat', { timeout: 15000 })
    step('返回聊天页', true)

    await browser.close()
    const pass = results.filter(Boolean).length
    console.log(`\n===== Admin UI 验证: ${pass}/${results.length} 通过 =====`)
    process.exit(pass === results.length ? 0 : 1)
  } catch (e) {
    await page.screenshot({ path: 'ui-admin-fail.png' })
    console.error('❌ 脚本异常:', e.message)
    await browser.close()
    process.exit(1)
  }
}

main()
