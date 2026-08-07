<template>
  <el-container class="layout">
    <!-- 左侧栏：品牌 + 新建会话 + 历史会话列表 + 用户信息 -->
    <el-aside width="260px" class="sidebar">
      <div class="brand">🤖 智能客服</div>
      <el-button type="primary" class="new-session" @click="onNewSession">
        ＋ 新建会话
      </el-button>
      <div class="session-list">
        <div
          v-for="s in sessions"
          :key="s.session_id"
          class="session-item"
          :class="{ active: s.session_id === currentSessionId }"
          @click="onSelectSession(s.session_id)"
        >
          <div class="session-info">
            <div class="session-title">{{ s.title }}</div>
            <div class="session-time">{{ formatTime(s.updated_at) }}</div>
          </div>
          <el-popconfirm
            title="删除后不可恢复，确认删除该会话？"
            width="220"
            @confirm="onDeleteSession(s.session_id)"
          >
            <template #reference>
              <el-button
                class="session-del"
                link
                type="danger"
                size="small"
                title="删除会话"
                @click.stop
              >
                删除
              </el-button>
            </template>
          </el-popconfirm>
        </div>
        <div v-if="sessions.length === 0" class="session-empty">暂无历史会话</div>
      </div>
      <div class="user-bar">
        <el-tag v-if="isAdmin" type="warning" size="small">管理员</el-tag>
        <span class="username">{{ user?.username }}</span>
        <el-button v-if="isAdmin" link type="primary" @click="router.push('/admin')">管理后台</el-button>
        <el-button link type="danger" @click="onLogout">退出</el-button>
      </div>
    </el-aside>

    <!-- 右侧聊天区 -->
    <el-main class="main">
      <div class="chat-wrap">
        <ChatPanel :messages="messages" :sending="sending" :progress="progress" />
        <div v-if="confirmVisible" class="confirm-bar">
          <span>请确认操作：</span>
          <el-button type="primary" :loading="sending" @click="sendConfirm">确认提交</el-button>
          <el-button :disabled="sending" @click="onSend('取消')">取消</el-button>
        </div>
        <ChatInput :disabled="sending" @send="onSend" />
      </div>
    </el-main>
  </el-container>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'

import type { SessionItem } from '@/api/sessionApi'
import ChatInput from '@/components/chat/ChatInput.vue'
import ChatPanel from '@/components/chat/ChatPanel.vue'
import { useAuth } from '@/composables/useAuth'
import { useChat } from '@/composables/useChat'
import { useSession } from '@/composables/useSession'
import { formatTime } from '@/utils/formatTime'

const { user, isAdmin, logout } = useAuth()
const { messages, sending, progress, confirmVisible, send, sendConfirm, clear, loadHistory } = useChat()
const { currentSessionId, restoreSession, setSession, clearSession, isBlankNewSession, loadSessions, removeSession } = useSession()
const router = useRouter()

const sessions = ref<SessionItem[]>([])

async function refreshSessions() {
  try {
    const data = await loadSessions()
    sessions.value = data.items || []
  } catch {
    sessions.value = []
  }
}

async function loadHistorySafely(sid: string) {
  try {
    await loadHistory(sid)
  } catch {
    // 会话已被删除 / 加载失败：降级为空白对话，不阻塞 UI
    clear()
    console.warn(`[session] 加载历史失败: ${sid}`)
  }
}

async function restoreActiveSession() {
  // 恢复分派：
  // 1) 当前会话有效（localStorage 恢复或用户刚切换）→ 尊重之，不覆盖
  // 2) localStorage 有值但会话已被删除/过期 → 回退列表第一条
  // 3) 首次进入（无任何记录）→ 恢复最近会话
  // 4) 主动新建会话后刷新（key 存在但空）→ 保持空白，不覆盖 localStorage
  let target: string | null = null
  if (currentSessionId.value && sessions.value.some((s) => s.session_id === currentSessionId.value)) {
    target = currentSessionId.value
  } else if (currentSessionId.value) {
    if (sessions.value.length) {
      target = sessions.value[0].session_id
      setSession(target)
    }
  } else if (sessions.value.length && !isBlankNewSession()) {
    target = sessions.value[0].session_id
    setSession(target)
  }
  if (target) await loadHistorySafely(target)
}

async function onSelectSession(sid: string) {
  if (sid === currentSessionId.value) return
  setSession(sid)
  await loadHistorySafely(sid)
}

function onNewSession() {
  clear()
  clearSession()
  // 不立即建会话：等用户首条消息时 ensureSession 创建，避免列表出现空会话
  void refreshSessions()
}

async function onDeleteSession(sid: string) {
  try {
    await removeSession(sid)
  } catch {
    console.warn(`[session] 删除失败: ${sid}`)
    return
  }
  await refreshSessions()
  if (sid === currentSessionId.value) {
    clear()
    clearSession()
    if (sessions.value.length) {
      setSession(sessions.value[0].session_id)
      await loadHistorySafely(sessions.value[0].session_id)
    }
  }
}

async function onSend(content: string) {
  await send(content)
  void refreshSessions() // 新会话 / 标题变化后同步列表
}

function onLogout() {
  logout()
  router.push('/login')
}

onMounted(async () => {
  restoreSession()
  await refreshSessions()
  await restoreActiveSession()
})
</script>

<style scoped>
.layout {
  height: 100vh;
}
.sidebar {
  display: flex;
  flex-direction: column;
  background: #fff;
  border-right: 1px solid #e4e7ed;
}
.brand {
  padding: 18px 16px;
  font-size: 17px;
  font-weight: 600;
}
.new-session {
  margin: 0 12px 12px;
}
.session-list {
  flex: 1;
  overflow-y: auto;
  padding: 0 8px;
}
.session-empty {
  padding: 16px 8px;
  font-size: 13px;
  color: #909399;
  text-align: center;
}
.session-item {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 10px 12px;
  margin-bottom: 4px;
  border-radius: 6px;
  cursor: pointer;
}
.session-item:hover {
  background: #f5f7fa;
}
.session-item.active {
  background: #ecf5ff;
}
.session-info {
  flex: 1;
  min-width: 0;
}
.session-title {
  font-size: 13px;
  color: #303133;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.session-time {
  margin-top: 4px;
  font-size: 12px;
  color: #909399;
}
.session-del {
  flex-shrink: 0;
}
.user-bar {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px 16px;
  border-top: 1px solid #e4e7ed;
}
.username {
  flex: 1;
  font-size: 13px;
  color: #606266;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.main {
  display: flex;
  flex-direction: column;
  padding: 0;
  background: #f5f7fa;
}
.chat-wrap {
  display: flex;
  flex-direction: column;
  height: 100%;
}
.confirm-bar {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 16px;
  background: #fff7e6;
  border-top: 1px solid #ffd591;
  font-size: 13px;
  color: #b88230;
}
</style>
