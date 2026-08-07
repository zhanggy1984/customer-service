<template>
  <div class="admin-page">
    <div class="admin-header">
      <span class="title">🛠 管理后台</span>
      <el-button link @click="router.push('/chat')">← 返回聊天</el-button>
    </div>

    <el-tabs v-model="tab">
      <!-- ========== 知识库管理 ========== -->
      <el-tab-pane label="知识库管理" name="kb">
        <el-card class="mb">
          <el-form :model="kbForm" label-width="80px">
            <el-form-item label="文档标题">
              <el-input v-model="kbForm.title" placeholder="如：运费政策" />
            </el-form-item>
            <el-form-item label="文档内容">
              <el-input v-model="kbForm.content" type="textarea" :rows="5" placeholder="粘贴 Markdown 内容" />
            </el-form-item>
            <el-button type="primary" :loading="kbLoading" @click="onUploadKb">上传知识库</el-button>
            <el-button :loading="syncLoading" @click="onSyncKb">同步知识库</el-button>
            <span class="kb-tip">同步：异常恢复后从源数据重建全部文档向量（含孤儿清理）</span>
          </el-form>
        </el-card>

        <el-table :data="kbItems" border>
          <el-table-column prop="source" label="文档标题" min-width="160" />
          <el-table-column prop="chunk_count" label="块数" width="70" />
          <el-table-column label="同步状态" width="100">
            <template #default="{ row }">
              <el-tag :type="row.sync_status === 'ok' ? 'success' : 'warning'" size="small">
                {{ row.sync_status === 'ok' ? '已同步' : '待同步' }}
              </el-tag>
            </template>
          </el-table-column>
          <el-table-column prop="updated_at" label="更新时间" width="170" />
          <el-table-column prop="updated_by" label="操作人" width="90" />
          <el-table-column label="操作" width="130">
            <template #default="{ row }">
              <el-button link type="primary" @click="openEditKb(row)">编辑</el-button>
              <el-button link type="danger" @click="onDeleteKb(row.source)">删除</el-button>
            </template>
          </el-table-column>
        </el-table>

        <!-- 编辑文档（内容级更新，MySQL 源数据 + ChromaDB 重建） -->
        <el-dialog v-model="editKbDialog" title="编辑文档" width="640px">
          <el-form label-width="80px">
            <el-form-item label="文档标题">{{ editKbForm.source }}</el-form-item>
            <el-form-item label="文档内容">
              <el-input v-model="editKbForm.content" type="textarea" :rows="10" />
            </el-form-item>
          </el-form>
          <template #footer>
            <el-button @click="editKbDialog = false">取消</el-button>
            <el-button type="primary" :loading="kbLoading" @click="onSaveEditKb">保存</el-button>
          </template>
        </el-dialog>
      </el-tab-pane>

      <!-- ========== 订单管理 ========== -->
      <el-tab-pane label="订单管理" name="orders">
        <div class="mb">
          <el-button type="primary" @click="openCreate">＋ 新建订单</el-button>
          <el-button type="danger" plain :loading="resetLoading" @click="onResetDemo">重置测试数据</el-button>
          <span class="kb-tip">重置：清空退货/退款/工单，恢复种子订单</span>
        </div>
        <el-table :data="orders" border>
          <el-table-column prop="order_id" label="订单号" width="180" />
          <el-table-column prop="user_id" label="用户" width="70" />
          <el-table-column label="状态" width="110">
            <template #default="{ row }">{{ statusText(ORDER_STATUS_MAP, row.status) }}</template>
          </el-table-column>
          <el-table-column prop="total_amount" label="金额" width="100" />
          <el-table-column prop="created_at" label="创建时间" width="170" />
          <el-table-column label="商品" min-width="200" show-overflow-tooltip>
            <template #default="{ row }">
              {{ (row.items || []).map((i: any) => `${i.name}×${i.quantity}(${statusText(ORDER_ITEM_STATUS_MAP, i.status)})`).join('、') }}
            </template>
          </el-table-column>
          <el-table-column label="操作" width="150">
            <template #default="{ row }">
              <el-button link type="primary" @click="openEdit(row)">改状态</el-button>
              <el-button link type="danger" @click="onDeleteOrder(row.order_id)">删除</el-button>
            </template>
          </el-table-column>
        </el-table>

        <!-- 新建订单 -->
        <el-dialog v-model="orderDialog" title="新建订单" width="1000px" top="6vh">
          <el-form :model="orderForm" label-width="80px" class="order-form">
            <el-row :gutter="16">
              <el-col :span="10">
                <el-form-item label="订单号"><el-input v-model="orderForm.order_id" placeholder="ORD-20240807-007" /></el-form-item>
              </el-col>
              <el-col :span="6">
                <el-form-item label="用户ID"><el-input-number v-model="orderForm.user_id" :min="1" style="width: 100%" /></el-form-item>
              </el-col>
              <el-col :span="8">
                <el-form-item label="状态">
                  <el-select v-model="orderForm.status" style="width: 100%">
                    <el-option v-for="s in ORDER_STATUS" :key="s" :label="statusText(ORDER_STATUS_MAP, s)" :value="s" />
                  </el-select>
                </el-form-item>
              </el-col>
            </el-row>
            <el-row :gutter="16">
              <el-col :span="8">
                <el-form-item label="总金额"><el-input-number v-model="orderForm.total_amount" :min="0" :precision="2" style="width: 100%" /></el-form-item>
              </el-col>
              <el-col :span="16">
                <el-form-item label="收货地址"><el-input v-model="orderForm.shipping_address" placeholder="收件人 / 收货地址" /></el-form-item>
              </el-col>
            </el-row>
            <el-form-item label="商品明细">
              <!-- el-table 天然列对齐，比 el-row/el-col 表头更可靠 -->
              <el-table :data="orderForm.items" border size="default" class="item-table">
                <el-table-column label="SKU 编号" width="150">
                  <template #default="{ row }">
                    <el-input v-model="row.item_id" placeholder="SKU-001" />
                  </template>
                </el-table-column>
                <el-table-column label="商品名称" min-width="240">
                  <template #default="{ row }">
                    <el-input v-model="row.name" placeholder="商品名称" />
                  </template>
                </el-table-column>
                <el-table-column label="单价" width="160">
                  <template #default="{ row }">
                    <el-input-number v-model="row.price" :min="0" :precision="2" controls-position="right" style="width: 100%" />
                  </template>
                </el-table-column>
                <el-table-column label="数量" width="150">
                  <template #default="{ row }">
                    <el-input-number v-model="row.quantity" :min="1" controls-position="right" style="width: 100%" />
                  </template>
                </el-table-column>
                <el-table-column label="可退" width="70" align="center">
                  <template #default="{ row }">
                    <el-checkbox v-model="row.returnable" />
                  </template>
                </el-table-column>
                <el-table-column label="操作" width="76" align="center">
                  <template #default="{ $index }">
                    <el-button link type="danger" @click="orderForm.items.splice($index, 1)">删除</el-button>
                  </template>
                </el-table-column>
              </el-table>
              <el-button link type="primary" @click="addItemRow" class="add-item-btn">＋ 添加商品</el-button>
            </el-form-item>
          </el-form>
          <template #footer>
            <el-button @click="orderDialog = false">取消</el-button>
            <el-button type="primary" :loading="orderLoading" @click="onCreateOrder">创建</el-button>
          </template>
        </el-dialog>

        <!-- 改状态 -->
        <el-dialog v-model="editDialog" title="修改订单" width="460px">
          <el-form label-width="80px">
            <el-form-item label="订单号">
              <el-input :model-value="editForm.order_id" disabled />
            </el-form-item>
            <el-form-item label="状态">
              <el-select v-model="editForm.status" style="width: 100%">
                <el-option v-for="s in ORDER_STATUS" :key="s" :label="statusText(ORDER_STATUS_MAP, s)" :value="s" />
              </el-select>
            </el-form-item>
          </el-form>
          <template #footer>
            <el-button @click="editDialog = false">取消</el-button>
            <el-button type="primary" @click="onUpdateOrder">保存</el-button>
          </template>
        </el-dialog>
      </el-tab-pane>
    </el-tabs>
  </div>
</template>

<script setup lang="ts">
import { ElMessage, ElMessageBox } from 'element-plus'
import { onMounted, reactive, ref } from 'vue'
import { useRouter } from 'vue-router'

import {
  createOrder,
  deleteKnowledge,
  deleteOrder,
  listKnowledge,
  listOrders,
  resetDemoData,
  syncKnowledge,
  updateKnowledge,
  updateOrder,
  uploadKnowledge,
  type KnowledgeDoc,
  type Order,
} from '@/api/adminApi'
import {
  ORDER_ITEM_STATUS_MAP,
  ORDER_STATUS_MAP,
  ORDER_STATUS_OPTIONS,
  statusText,
} from '@/constants/status'

const ORDER_STATUS = ORDER_STATUS_OPTIONS.map((o) => o.value)

const router = useRouter()
const tab = ref('kb')

// ---------- 知识库 ----------
const kbItems = ref<KnowledgeDoc[]>([])
const kbLoading = ref(false)
const syncLoading = ref(false)
const kbForm = reactive({ title: '', content: '' })
const editKbDialog = ref(false)
const editKbForm = reactive({ source: '', content: '' })

async function loadKb() {
  const data = await listKnowledge()
  kbItems.value = data.items
}

async function onUploadKb() {
  if (!kbForm.title.trim() || !kbForm.content.trim()) {
    ElMessage.warning('请填写标题和内容')
    return
  }
  kbLoading.value = true
  try {
    await uploadKnowledge(kbForm.title.trim(), kbForm.content)
    ElMessage.success('上传成功')
    kbForm.title = ''
    kbForm.content = ''
    await loadKb()
  } catch (e: any) {
    ElMessage.error(e.response?.data?.detail || '上传失败')
  } finally {
    kbLoading.value = false
  }
}

function openEditKb(row: KnowledgeDoc) {
  editKbForm.source = row.source
  editKbForm.content = row.content
  editKbDialog.value = true
}

async function onSaveEditKb() {
  if (!editKbForm.content.trim()) {
    ElMessage.warning('内容不能为空')
    return
  }
  kbLoading.value = true
  try {
    await updateKnowledge(editKbForm.source, editKbForm.content)
    ElMessage.success('已保存')
    editKbDialog.value = false
    await loadKb()
  } catch (e: any) {
    ElMessage.error(e.response?.data?.detail || '保存失败')
  } finally {
    kbLoading.value = false
  }
}

async function onDeleteKb(source: string) {
  await deleteKnowledge(source)
  ElMessage.success('已删除')
  await loadKb()
}

async function onSyncKb() {
  syncLoading.value = true
  try {
    const r = await syncKnowledge()
    ElMessage.success(`同步完成：重建 ${r.synced} 篇，清理孤儿 ${r.orphan_removed} 个`)
    await loadKb()
  } catch (e: any) {
    ElMessage.error(e.response?.data?.detail || '同步失败')
  } finally {
    syncLoading.value = false
  }
}

// ---------- 订单 ----------
const orders = ref<Order[]>([])
const orderDialog = ref(false)
const editDialog = ref(false)
const orderLoading = ref(false)
const resetLoading = ref(false)

interface OrderItemForm {
  item_id: string
  name: string
  price: number
  quantity: number
  returnable: boolean
}

const orderForm = reactive({
  order_id: '',
  user_id: 1,
  status: 'PAID',
  total_amount: 0,
  shipping_address: '',
  items: [] as OrderItemForm[],
})

const editForm = reactive({ order_id: '', status: 'PAID' })

async function loadOrders() {
  const data = await listOrders()
  orders.value = data.items
}

function addItemRow() {
  orderForm.items.push({ item_id: '', name: '', price: 0, quantity: 1, returnable: true })
}

function openCreate() {
  orderForm.order_id = ''
  orderForm.user_id = 1
  orderForm.status = 'PAID'
  orderForm.total_amount = 0
  orderForm.shipping_address = ''
  orderForm.items = []
  addItemRow()
  orderDialog.value = true
}

async function onCreateOrder() {
  if (!orderForm.order_id.trim() || orderForm.items.length === 0) {
    ElMessage.warning('请填写订单号和商品明细')
    return
  }
  orderLoading.value = true
  try {
    await createOrder({
      order_id: orderForm.order_id.trim(),
      user_id: orderForm.user_id,
      status: orderForm.status,
      total_amount: orderForm.total_amount,
      shipping_address: orderForm.shipping_address,
      items: orderForm.items,
    })
    ElMessage.success('已创建')
    orderDialog.value = false
    await loadOrders()
  } catch (e: any) {
    ElMessage.error(e.response?.data?.detail || '创建失败')
  } finally {
    orderLoading.value = false
  }
}

function openEdit(row: Order) {
  editForm.order_id = row.order_id
  editForm.status = row.status
  editDialog.value = true
}

async function onUpdateOrder() {
  await updateOrder(editForm.order_id, { status: editForm.status })
  ElMessage.success('已更新')
  editDialog.value = false
  await loadOrders()
}

async function onDeleteOrder(orderId: string) {
  await deleteOrder(orderId)
  ElMessage.success('已删除')
  await loadOrders()
}

async function onResetDemo() {
  // 破坏性操作：二次确认
  try {
    await ElMessageBox.confirm(
      '将清空全部退货单、退款单、投诉工单，并恢复 5 个种子订单及商品明细。确定重置吗？',
      '重置测试数据',
      { type: 'warning', confirmButtonText: '确认重置', cancelButtonText: '取消' },
    )
  } catch {
    return // 用户取消
  }
  resetLoading.value = true
  try {
    const r = await resetDemoData()
    ElMessage.success(`已重置：恢复 ${r.orders} 个种子订单`)
    await loadOrders()
  } catch (e: any) {
    ElMessage.error(e.response?.data?.detail || '重置失败')
  } finally {
    resetLoading.value = false
  }
}

onMounted(async () => {
  await loadKb()
  await loadOrders()
})
</script>

<style scoped>
.admin-page {
  padding: 20px;
  max-width: 1100px;
  margin: 0 auto;
}
.admin-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}
.title {
  font-size: 18px;
  font-weight: 600;
}
.mb {
  margin-bottom: 16px;
}
.item-table {
  margin-bottom: 8px;
}
.order-form {
  max-height: 62vh;
  overflow-y: auto;
  padding-right: 6px;
}
.add-item-btn {
  margin-top: 2px;
}
.kb-tip {
  margin-left: 12px;
  color: #909399;
  font-size: 12px;
}
</style>
