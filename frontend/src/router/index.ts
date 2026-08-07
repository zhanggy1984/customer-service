/** 路由 + 登录守卫 */
import { createRouter, createWebHistory } from 'vue-router'

import { useAuthStore } from '@/stores/authStore'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', redirect: '/chat' },
    { path: '/login', name: 'login', component: () => import('@/views/LoginView.vue'), meta: { public: true } },
    { path: '/register', name: 'register', component: () => import('@/views/RegisterView.vue'), meta: { public: true } },
    { path: '/chat', name: 'chat', component: () => import('@/views/CustomerServiceView.vue') },
    { path: '/admin', name: 'admin', component: () => import('@/views/AdminView.vue'), meta: { requiresAdmin: true } },
  ],
})

router.beforeEach((to) => {
  const auth = useAuthStore()
  if (!to.meta.public && !auth.isLoggedIn) {
    // 未登录访问受保护页面 → 登录页
    return { path: '/login', query: { redirect: to.fullPath } }
  }
  if (to.meta.requiresAdmin && !auth.isAdmin) {
    // 非 admin 访问管理后台 → 聊天页
    return { path: '/' }
  }
  if (to.meta.public && auth.isLoggedIn) {
    // 已登录访问登录/注册页 → 聊天页
    return { path: '/' }
  }
})

export default router
