<template>
  <div ref="listEl" class="chat-panel">
    <div v-if="messages.length === 0" class="empty">
      您好，我是智能客服，请问有什么可以帮您？
    </div>

    <div v-for="m in messages" :key="m.id" class="msg-row" :class="m.role">
      <div class="bubble">
        {{ m.content }}
        <span v-if="m.streaming" class="cursor">▌</span>
      </div>
    </div>

    <!-- 进度条（Agent 阶段事件驱动） -->
    <div v-if="progress" class="progress-bar">
      <span class="spinner"></span>
      {{ progress }}
    </div>
  </div>
</template>

<script setup lang="ts">
import { nextTick, ref, watch } from 'vue'

import type { ChatMessage } from '@/composables/useChat'

const props = defineProps<{ messages: ChatMessage[]; sending: boolean; progress: string }>()

const listEl = ref<HTMLElement>()

watch(
  () => `${props.messages.length}-${props.progress}`,
  async () => {
    await nextTick()
    if (listEl.value) listEl.value.scrollTop = listEl.value.scrollHeight
  },
)
</script>

<style scoped>
.chat-panel {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
  background: #f5f7fa;
}
.empty {
  text-align: center;
  color: #909399;
  margin-top: 40px;
}
.msg-row {
  display: flex;
  margin-bottom: 16px;
}
.msg-row.user {
  justify-content: flex-end;
}
.msg-row.assistant {
  justify-content: flex-start;
}
.bubble {
  max-width: 70%;
  padding: 10px 14px;
  border-radius: 8px;
  line-height: 1.6;
  white-space: pre-wrap;
  word-break: break-word;
  font-size: 14px;
}
.msg-row.user .bubble {
  background: #409eff;
  color: #fff;
  border-top-right-radius: 2px;
}
.msg-row.assistant .bubble {
  background: #fff;
  color: #303133;
  border-top-left-radius: 2px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
}
.cursor {
  color: #409eff;
  animation: blink 1s step-end infinite;
}
@keyframes blink {
  50% {
    opacity: 0;
  }
}
.progress-bar {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 8px 14px;
  background: #ecf5ff;
  border: 1px solid #d9ecff;
  border-radius: 8px;
  color: #409eff;
  font-size: 13px;
}
.spinner {
  width: 14px;
  height: 14px;
  border: 2px solid #a0cfff;
  border-top-color: #409eff;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}
@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}
</style>
