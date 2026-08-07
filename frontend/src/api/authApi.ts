/** 认证相关接口 */
import client from './client'
import type { LoginRequest, RegisterRequest, TokenResponse } from '@/types'

export const loginApi = (data: LoginRequest) =>
  client.post('/auth/login', data) as Promise<TokenResponse>

export const registerApi = (data: RegisterRequest) =>
  client.post('/auth/register', data) as Promise<{ msg: string }>
