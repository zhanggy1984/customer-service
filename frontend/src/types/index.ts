/** 前端类型定义 */

export interface UserInfo {
  id: number
  username: string
  role: 'admin' | 'user'
  phone: string | null
}

export interface LoginRequest {
  username: string
  password: string
}

export interface RegisterRequest {
  username: string
  password: string
  phone?: string
}

export interface TokenResponse {
  access_token: string
  token_type: string
  user: UserInfo
}
