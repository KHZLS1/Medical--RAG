import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// 后端地址: http://localhost:8000
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: true,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        // 关键：确保 SSE 流式响应不被缓冲
        configure: (proxy, _options) => {
          proxy.on('proxyReq', (proxyReq, _req, res) => {
            // 告诉代理服务器不要缓冲 SSE 响应
            res.setHeader('X-Accel-Buffering', 'no')
            res.setHeader('Cache-Control', 'no-cache, no-transform')
          })
          proxy.on('proxyRes', (proxyRes, _req, res) => {
            // 如果是 SSE 响应，确保流式透传
            const ct = proxyRes.headers['content-type']
            if (ct && (ct as string).includes('text/event-stream')) {
              res.setHeader('Cache-Control', 'no-cache, no-transform')
              res.setHeader('Connection', 'keep-alive')
              res.setHeader('X-Accel-Buffering', 'no')
              proxyRes.headers['cache-control'] = 'no-cache, no-transform'
              proxyRes.headers['x-accel-buffering'] = 'no'
            }
          })
        },
      },
    },
  },
})
