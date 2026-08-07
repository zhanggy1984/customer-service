<template>
  <div class="auth-page">
    <el-card class="auth-card">
      <h2 class="title">注册账号</h2>
      <p class="subtitle">创建你的客服账号</p>
      <el-form label-position="top" @keyup.enter="onSubmit">
        <el-form-item label="用户名">
          <el-input v-model="form.username" placeholder="3-32 个字符" clearable />
        </el-form-item>
        <el-form-item label="密码">
          <el-input v-model="form.password" type="password" placeholder="至少 6 位" show-password />
        </el-form-item>
        <el-form-item label="确认密码">
          <el-input v-model="form.confirm" type="password" placeholder="再次输入密码" show-password />
        </el-form-item>
        <el-form-item label="手机号（选填）">
          <el-input v-model="form.phone" placeholder="用于接收售后通知" clearable />
        </el-form-item>
        <el-button type="primary" class="submit" :loading="loading" @click="onSubmit">
          注 册
        </el-button>
      </el-form>
      <div class="footer">
        已有账号？
        <router-link to="/login">去登录</router-link>
      </div>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ElMessage } from 'element-plus'
import { reactive, ref } from 'vue'
import { useRouter } from 'vue-router'

import { useAuthStore } from '@/stores/authStore'

const auth = useAuthStore()
const router = useRouter()

const loading = ref(false)
const form = reactive({ username: '', password: '', confirm: '', phone: '' })

async function onSubmit() {
  const name = form.username.trim()
  if (name.length < 3 || name.length > 32) {
    ElMessage.warning('用户名长度需为 3-32 个字符')
    return
  }
  if (form.password.length < 6) {
    ElMessage.warning('密码至少 6 位')
    return
  }
  if (form.password !== form.confirm) {
    ElMessage.warning('两次输入的密码不一致')
    return
  }
  loading.value = true
  try {
    await auth.register({ username: name, password: form.password, phone: form.phone || undefined })
    ElMessage.success('注册成功，请登录')
    router.push('/login')
  } catch (e: any) {
    ElMessage.error(e.response?.data?.detail || '注册失败，请重试')
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
