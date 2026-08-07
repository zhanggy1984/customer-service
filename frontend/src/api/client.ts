/** Axios 实例：自动带 Bearer token；401 时清理登录态并跳回登录页 */
import axios from 'axios'

import { useAuthStore } from '@/stores/authStore'

const client = axios.create({
  baseURL: '/api/v1',
  timeout: 30_000,
})

// 请求拦截器：注入 token
client.interceptors.request.use((config) => {
  const auth = useAuthStore()
  if (auth.token) {
    config.headers.Authorization = `Bearer ${auth.token}`
  }
  return config
})

// 响应拦截器：直接返回 data；401 统一处理
client.interceptors.response.use(
  (resp) => resp.data,
  (error) => {
    if (error.response?.status === 401) {
      const auth = useAuthStore()
      auth.logout()
      // 用整页跳转，避免引入 router 造成模块循环依赖
      if (window.location.pathname !== '/login') {
        window.location.href = '/login'
      }
    }
    return Promise.reject(error)
  },
)

export default client
