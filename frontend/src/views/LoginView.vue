<template>
  <div class="auth-page">
    <el-card class="auth-card">
      <h2 class="title">智能客服</h2>
      <p class="subtitle">请登录后继续</p>
      <el-form label-position="top" @keyup.enter="onSubmit">
        <el-form-item label="用户名">
          <el-input v-model="form.username" placeholder="请输入用户名" clearable />
        </el-form-item>
        <el-form-item label="密码">
          <el-input v-model="form.password" type="password" placeholder="请输入密码" show-password />
        </el-form-item>
        <el-button type="primary" class="submit" :loading="loading" @click="onSubmit">
          登 录
        </el-button>
      </el-form>
      <div class="footer">
        还没有账号？
        <router-link to="/register">立即注册</router-link>
      </div>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ElMessage } from 'element-plus'
import { reactive, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'

import { useAuthStore } from '@/stores/authStore'

const auth = useAuthStore()
const router = useRouter()
const route = useRoute()

const loading = ref(false)
const form = reactive({ username: '', password: '' })

async function onSubmit() {
  if (!form.username.trim() || !form.password) {
    ElMessage.warning('请输入用户名和密码')
    return
  }
  loading.value = true
  try {
    await auth.login({ username: form.username.trim(), password: form.password })
    ElMessage.success('登录成功')
    router.push((route.query.redirect as string) || '/chat')
  } catch (e: any) {
    ElMessage.error(e.response?.data?.detail || '登录失败，请重试')
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.auth-page {
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}
.auth-card {
  width: 380px;
  padding: 8px 12px;
}
.title {
  text-align: center;
  margin: 0;
}
.subtitle {
  text-align: center;
  color: #909399;
  margin: 4px 0 16px;
  font-size: 13px;
}
.submit {
  width: 100%;
  margin-top: 4px;
}
.footer {
  text-align: center;
  margin-top: 14px;
  font-size: 13px;
  color: #606266;
}
</style>
