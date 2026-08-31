/** 认证相关接口 */
import axios from 'axios'
import type { LoginRequest, RegisterRequest, TokenResponse } from '@/types'

// T15：登录/注册统一走 /api/auth/*（4 家 agent 契约一致），不经过业务 baseURL(/api/v1)
const authClient = axios.create({ baseURL: '/api', timeout: 30_000 })

export const loginApi = (data: LoginRequest) =>
  authClient.post('/auth/login', data).then((r) => r.data) as Promise<TokenResponse>

export const registerApi = (data: RegisterRequest) =>
  authClient.post('/auth/register', data).then((r) => r.data) as Promise<{ msg: string }>
