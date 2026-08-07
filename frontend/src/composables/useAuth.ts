/** 登录态组合式封装（基于 authStore） */
import { storeToRefs } from 'pinia'

import { useAuthStore } from '@/stores/authStore'

export function useAuth() {
  const auth = useAuthStore()
  const { user, isLoggedIn, isAdmin } = storeToRefs(auth)
  return {
    user,
    isLoggedIn,
    isAdmin,
    login: auth.login,
    register: auth.register,
    logout: auth.logout,
  }
}
