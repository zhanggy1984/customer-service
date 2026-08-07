<template>
  <div class="chat-input">
    <el-input
      v-model="text"
      type="textarea"
      :rows="2"
      :disabled="disabled"
      resize="none"
      placeholder="请输入您的问题（Enter 发送，Shift+Enter 换行）"
      @keydown.enter.exact.prevent="onSend"
    />
    <el-button type="primary" class="send-btn" :loading="disabled" @click="onSend">
      发送
    </el-button>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'

const props = defineProps<{ disabled: boolean }>()
const emit = defineEmits<{ (e: 'send', content: string): void }>()

const text = ref('')

function onSend() {
  const t = text.value.trim()
  if (!t || props.disabled) return
  emit('send', t)
  text.value = ''
}
</script>

<style scoped>
.chat-input {
  display: flex;
  gap: 12px;
  padding: 12px 16px;
  border-top: 1px solid #e4e7ed;
  background: #fff;
}
.send-btn {
  height: 52px;
}
</style>
