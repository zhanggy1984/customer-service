<template>
  <div ref="listEl" class="chat-panel">
    <div v-if="messages.length === 0" class="empty">
      您好，我是智能客服，请问有什么可以帮您？
    </div>

    <div v-for="m in messages" :key="m.id" class="msg-row" :class="m.role">
      <div class="msg-col">
        <!-- 思考过程折叠块：默认收起，点击展开全文；reasoning 事件实时累积，仅终答消息有值 -->
        <div v-if="m.reasoning" class="reasoning">
          <button class="reasoning-toggle" @click="toggleReasoning(m.id)">
            💭 思考过程 {{ expandedReasoning.has(m.id) ? '收起' : '展开' }}
          </button>
          <div v-if="expandedReasoning.has(m.id)" class="reasoning-content">{{ m.reasoning }}</div>
        </div>
        <!-- 检索来源列表：默认收起，点击展开全文；序号与回答中 [来源N] 对应 -->
        <div v-if="m.sources?.length" class="sources">
          <button class="sources-toggle" @click="toggleSources(m.id)">
            📎 来源（{{ m.sources.length }}） {{ expandedSources.has(m.id) ? '收起' : '展开' }}
          </button>
          <div v-if="expandedSources.has(m.id)" class="sources-body">
            <div v-for="(s, i) in m.sources" :key="i" class="source-item">
              <div class="source-line">
                <span class="source-ref">[来源{{ i + 1 }}]</span>
                <span class="source-path">{{ s.source }}</span>
              </div>
              <div class="source-text">{{ s.text }}</div>
            </div>
          </div>
        </div>
        <div class="bubble">
          {{ m.content }}
          <span v-if="m.streaming" class="cursor">▌</span>
        </div>
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
// 折叠状态：思考过程与来源列表各自独立，默认收起（Set 存已展开的消息 id）
const expandedReasoning = ref<Set<string>>(new Set())
const expandedSources = ref<Set<string>>(new Set())

function toggleReasoning(id: string) {
  const next = new Set(expandedReasoning.value)
  if (next.has(id)) next.delete(id)
  else next.add(id)
  expandedReasoning.value = next
}

function toggleSources(id: string) {
  const next = new Set(expandedSources.value)
  if (next.has(id)) next.delete(id)
  else next.add(id)
  expandedSources.value = next
}

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
.msg-col {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  max-width: 85%;
}
.msg-row.user .msg-col {
  align-items: flex-end;
}
.reasoning {
  margin-bottom: 8px;
}
.reasoning-toggle {
  background: none;
  border: 1px solid #d9ecff;
  color: #409eff;
  border-radius: 6px;
  padding: 3px 10px;
  font-size: 12px;
  cursor: pointer;
}
.reasoning-toggle:hover {
  border-color: #409eff;
}
.reasoning-content {
  margin-top: 6px;
  padding: 8px 10px;
  background: #f0f2f5;
  border-radius: 6px;
  font-size: 12px;
  color: #909399;
  line-height: 1.6;
  white-space: pre-wrap;
  word-break: break-word;
}
.sources {
  width: 100%;
  margin: 8px 0;
  font-size: 12px;
  color: #8a6d3b;
  line-height: 1.6;
}
.sources-toggle {
  background: #fdf6ec;
  border: 1px solid #f5dab1;
  color: #e6a23c;
  border-radius: 6px;
  padding: 3px 10px;
  font-size: 12px;
  cursor: pointer;
}
.sources-toggle:hover {
  border-color: #e6a23c;
}
.sources-body {
  margin-top: 6px;
  padding: 8px 10px;
  background: #fdf6ec;
  border: 1px solid #f5dab1;
  border-radius: 6px;
}
.source-item {
  margin-bottom: 6px;
}
.source-item:last-child {
  margin-bottom: 0;
}
.source-line {
  display: flex;
  gap: 6px;
  align-items: baseline;
}
.source-ref {
  color: #e6a23c;
  font-weight: 600;
  white-space: nowrap;
}
.source-path {
  word-break: break-all;
}
.source-text {
  color: #8a6d3b;
  white-space: pre-wrap;
  word-break: break-word;
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
