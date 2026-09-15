import { useEffect, useState } from 'react'
import Chat from './components/Chat'
import Sidebar from './components/Sidebar'
import DocumentPanel from './components/DocumentPanel'
import {
  getConversations,
  deleteConversation,
  getConversationMessages,
  checkHealth,
  type Conversation,
  type ChatMessageData,
} from './api/chat'

export default function App() {
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [activeConvId, setActiveConvId] = useState<number | null>(null)
  const [messages, setMessages] = useState<ChatMessageData[]>([])
  const [loadingMsgs, setLoadingMsgs] = useState(false)

  const [health, setHealth] = useState<string>('检测中...')
  const [showDocPanel, setShowDocPanel] = useState(false)

  async function refreshHealth() {
    try {
      const r = await checkHealth()
      setHealth(`Milvus: ${r.milvus_uri} | collection: ${r.collection}`)
    } catch {
      setHealth('后端未响应，请先启动 backend')
    }
  }

  async function loadConversations(keepActive = false) {
    try {
      const list = await getConversations()
      setConversations(list)
      // 仅在需要开新对话时重置选中状态；刷新列表时保留当前会话，以免破坏多轮上下文
      if (!keepActive) {
        setActiveConvId(null)
        setMessages([])
      }
    } catch (e) {
      console.error('加载会话列表失败:', e)
    }
  }

  async function selectConversation(id: number) {
    setActiveConvId(id)
    setLoadingMsgs(true)
    try {
      const msgs = await getConversationMessages(id)
      setMessages(msgs)
    } catch (e) {
      console.error('加载消息失败:', e)
      setMessages([])
    } finally {
      setLoadingMsgs(false)
    }
  }

  function handleCreate() {
    setActiveConvId(null)
    setMessages([])
  }

  async function handleDelete(id: number) {
    try {
      await deleteConversation(id)
      setConversations((prev) => prev.filter((c) => c.id !== id))
      if (activeConvId === id) {
        setActiveConvId(null)
        setMessages([])
      }
    } catch (e) {
      console.error('删除会话失败:', e)
    }
  }

  function handleConversationCreated(id: number) {
    setActiveConvId(id)
    loadConversations(true)   // 保留刚创建的会话，避免被清空成新对话
  }

  useEffect(() => {
    refreshHealth()
    loadConversations()
  }, [])

  return (
    <div className="app-layout">
      <Sidebar
        conversations={conversations}
        activeId={activeConvId}
        onSelect={selectConversation}
        onCreate={handleCreate}
        onDelete={handleDelete}
      />
      <div className="app">
        <header className="header">
          <div className="header-left">
            <h1>🏥 医疗智能问答助手</h1>
            <span className="status">{health}</span>
          </div>
          <div className="header-right">
            <button onClick={refreshHealth} className="btn-small">
              刷新状态
            </button>
            <button
              onClick={() => setShowDocPanel(true)}
              className="btn-primary"
            >
              📄 文档管理
            </button>
          </div>
        </header>

        <div className="disclaimer">
          ⚠️ 本系统回答仅基于公开医学问答数据集与大模型生成，仅供参考，<b>不能替代执业医师诊断</b>。
          涉及急症请立即拨打 120。
        </div>

        {loadingMsgs ? (
          <div className="chat-loading">加载历史消息...</div>
        ) : (
          <Chat
            conversationId={activeConvId}
            initialMessages={messages}
            onConversationCreated={handleConversationCreated}
          />
        )}
      </div>

      {showDocPanel && <DocumentPanel onClose={() => setShowDocPanel(false)} />}
    </div>
  )
}
