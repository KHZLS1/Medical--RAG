import React, { useEffect, useState } from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import Screen from './components/Screen'
import './index.css'

/**
 * 顶层路由：只有两个页面，所以用 hash 而不是引 react-router。
 *
 * `#/screen` 是展厅数据大屏 —— 它必须能**独立成页**（挂在墙上的机器直接开这个
 * 地址就能轮播），所以在这里分流，而不是塞进 App 里当一层弹窗：走 App 的话
 * 会话列表、健康检查那些请求会跟着一起发，展厅机并不需要。
 *
 * hash 而非 path：静态托管下不需要服务端 rewrite，双击 HTML 也能用。
 */
function Root() {
  const [screen, setScreen] = useState(() => window.location.hash === '#/screen')

  useEffect(() => {
    function onHash() {
      setScreen(window.location.hash === '#/screen')
    }
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])

  /**
   * 退出大屏时直接改状态 + 抹掉 hash，而不是 `location.hash = ''`。
   * 后者会留下一个孤零零的 `#`，且必须依赖 hashchange 事件绕一圈才生效。
   */
  function exitScreen() {
    window.history.replaceState(
      null,
      '',
      window.location.pathname + window.location.search,
    )
    setScreen(false)
  }

  if (screen) return <Screen onExit={exitScreen} />
  return <App />
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <Root />
  </React.StrictMode>,
)