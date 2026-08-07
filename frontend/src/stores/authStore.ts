/** 登录态：token + 用户信息，localStorage 持久化（刷新不丢） */
import { defineStore } from 'pinia'
import { computed, ref } from 'vue'

import { loginApi, registerApi } from '@/api/authApi'
import type { LoginRequest, RegisterRequest, UserInfo } from '@/types'

const TOKEN_KEY = 'cs_token'
const USER_KEY = 'cs_user'

function readUser(): UserInfo | null {
  try {
    return JSON.parse(localStorage.getItem(USER_KEY) || 'null')
  } catch {
    return null
  }
}

export const useAuthStore = defineStore('auth', () => {
  const token = ref<string>(localStorage.getItem(TOKEN_KEY) || '')
  const user = ref<UserInfo | null>(readUser())

  const isLoggedIn = computed(() => !!token.value)
  const isAdmin = computed(() => user.value?.role === 'admin')

  async function login(payload: LoginRequest) {
    const data = await loginApi(payload)
    token.value = data.access_token
    user.value = data.user
    localStorage.setItem(TOKEN_KEY, data.access_token)
    localStorage.setItem(USER_KEY, JSON.stringify(data.user))
  }

  async function register(payload: RegisterRequest) {
    await registerApi(payload)
  }

  function logout() {
    token.value = ''
    user.value = null
    localStorage.removeItem(TOKEN_KEY)
    localStorage.removeItem(USER_KEY)
  }

  return { token, user, isLoggedIn, isAdmin, login, register, logout }
})
