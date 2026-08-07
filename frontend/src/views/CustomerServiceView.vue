<template>
  <el-container class="layout">
    <!-- 左侧栏：品牌 + 新建会话 + 用户信息 -->
    <el-aside width="260px" class="sidebar">
      <div class="brand">🤖 智能客服</div>
      <el-button type="primary" class="new-session" @click="onNewSession">
        ＋ 新建会话
      </el-button>
      <div class="session-list">
        <!-- Phase 1 单会话；多会话列表后续版本接入 -->
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
import { useRouter } from 'vue-router'

import ChatInput from '@/components/chat/ChatInput.vue'
import ChatPanel from '@/components/chat/ChatPanel.vue'
import { useAuth } from '@/composables/useAuth'
import { useChat } from '@/composables/useChat'
import { useSession } from '@/composables/useSession'

const { user, isAdmin, logout } = useAuth()
const { messages, sending, progress, confirmVisible, send, sendConfirm, clear } = useChat()
const { clearSession } = useSession()
const router = useRouter()

async function onSend(content: string) {
  await send(content)
}

function onNewSession() {
  clear()
  clearSession()
}

function onLogout() {
  logout()
  router.push('/login')
}
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
