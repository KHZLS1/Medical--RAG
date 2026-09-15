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
      },
    },
  },
})
