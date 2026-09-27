import { useEffect, useState } from 'react'
import Chat from './components/Chat'
import Rail from './components/Rail'
import Intro from './components/Intro'
import DocumentPanel from './components/DocumentPanel'
import FeedbackPanel from './components/FeedbackPanel'
import {
  getConversations,
  deleteConversation,
  getConversationMessages,
  checkHealth,
  type Conversation,
  type ChatMessageData,
} from './api/chat'

/**
 * 「当前在看哪个会话」的落盘位置。
 *
 * 为什么需要它：刷新页面后 React 组件全部重建，`activeConvId` 回到 null，
 * 主区域就退回欢迎页 —— 用户看到的"对话丢了"其实是**选中态没持久化**，
 * 消息一直好好躺在 MySQL 里。会话列表能靠 /api/conversations 恢复，
 * 唯独"当前选中哪个"这个纯前端状态没有后端来源，只能自己存。
 */
const ACTIVE_CONV_KEY = 'medical-rag.active-conversation-id'
const THEME_KEY = 'medical-rag.theme'
/**
 * 海报页「看过了」的标记。
 * 只认「有没有这个键」，不认值 —— 想给所有老用户重新看一次，改个键名即可，
 * 不用写数据迁移。
 */
const INTRO_KEY = 'medical-rag.intro-seen'

function readStoredConvId(): number | null {
  const raw = localStorage.getItem(ACTIVE_CONV_KEY)
  if (raw === null) return null
  const n = Number(raw)
  return Number.isInteger(n) && n > 0 ? n : null
}

function readStoredTheme(): 'light' | 'dark' {
  return localStorage.getItem(THEME_KEY) === 'dark' ? 'dark' : 'light'
}

// 在模块作用域就把主题写到 <html> 上：放到 useEffect 里的话，
// 首帧会先按浅色画一遍再跳成深色，深色用户每次刷新都要闪一下白。
document.documentElement.dataset.theme = readStoredTheme()

export default function App() {
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [activeConvId, setActiveConvId] = useState<number | null>(readStoredConvId)
  const [messages, setMessages] = useState<ChatMessageData[]>([])
  const [loadingMsgs, setLoadingMsgs] = useState(false)

  const [healthText, setHealthText] = useState('检测中…')
  const [healthOk, setHealthOk] = useState(true)
  const [healthChecking, setHealthChecking] = useState(false)
  const [healthAt, setHealthAt] = useState('')

  const [theme, setTheme] = useState<'light' | 'dark'>(readStoredTheme)
  const [navExpanded, setNavExpanded] = useState(false)
  const [evidenceOpen, setEvidenceOpen] = useState(true)
  const [showDocPanel, setShowDocPanel] = useState(false)
  const [showFbPanel, setShowFbPanel] = useState(false)
  const [showIntro, setShowIntro] = useState(() => localStorage.getItem(INTRO_KEY) === null)

  // 标记写在「关闭」时而不是「显示」时：这样看到一半刷新页面，海报还在，
  // 不会出现"刷新一下说明就再也找不到了"。
  function closeIntro() {
    localStorage.setItem(INTRO_KEY, '1')
    setShowIntro(false)
  }

  async function refreshHealth() {
    setHealthChecking(true)
    try {
      const r = await checkHealth()
      setHealthText(`Milvus: ${r.milvus_uri} · collection: ${r.collection}`)
      setHealthOk(true)
    } catch {
      setHealthText('后端未响应，请先启动 backend')
      setHealthOk(false)
    } finally {
      setHealthChecking(false)
      // 状态串成功/失败时都可能一字不变（Milvus 地址和 collection 不会变），
      // 只改文本的话点按钮等于没反应。记一个检查时刻，让每次刷新都有可见结果。
      setHealthAt(new Date().toLocaleTimeString('zh-CN', { hour12: false }))
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

  async function selectConversation(id: number): Promise<boolean> {
    setActiveConvId(id)
    setLoadingMsgs(true)
    try {
      const msgs = await getConversationMessages(id)
      setMessages(msgs)
      return true
    } catch (e) {
      console.error('加载消息失败:', e)
      setMessages([])
      return false
    } finally {
      setLoadingMsgs(false)
    }
  }

  function handleCreate() {
    setActiveConvId(null)
    setMessages([])
    setNavExpanded(false)
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

  // 选中态落盘：新建/切换/删除会话都会经过这里，不用在每个 handler 里各写一遍。
  useEffect(() => {
    if (activeConvId === null) localStorage.removeItem(ACTIVE_CONV_KEY)
    else localStorage.setItem(ACTIVE_CONV_KEY, String(activeConvId))
  }, [activeConvId])

  useEffect(() => {
    document.documentElement.dataset.theme = theme
    localStorage.setItem(THEME_KEY, theme)
  }, [theme])

  useEffect(() => {
    refreshHealth()
    // 必须 keepActive=true：默认的 loadConversations() 会把 activeConvId 清成 null，
    // 那正是"刷新后回到欢迎页"的成因。
    loadConversations(true)

    const stored = readStoredConvId()
    if (stored === null) return
    selectConversation(stored).then((ok) => {
      // 会话可能已被删除（或换了后端库）：恢复失败就退回新对话，
      // 否则会留一个发不出去的 stale id，之后每条消息都 404。
      if (!ok) setActiveConvId(null)
    })
  }, [])

  const activeConv = conversations.find((c) => c.id === activeConvId)
  const title = activeConv?.title || '新对话'

  return (
    <div
      className="app"
      data-nav={navExpanded ? 'expanded' : 'collapsed'}
      data-evidence={evidenceOpen ? 'open' : 'closed'}
    >
      <Rail
        conversations={conversations}
        activeId={activeConvId}
        expanded={navExpanded}
        onToggleExpanded={() => setNavExpanded((v) => !v)}
        onSelect={(id) => {
          selectConversation(id)
          setNavExpanded(false)
        }}
        onCreate={handleCreate}
        onDelete={handleDelete}
        theme={theme}
        onToggleTheme={() => setTheme((t) => (t === 'light' ? 'dark' : 'light'))}
        onOpenDocs={() => setShowDocPanel(true)}
        onOpenFeedback={() => setShowFbPanel(true)}
        onOpenIntro={() => setShowIntro(true)}
        onOpenScreen={() => {
          window.location.hash = '#/screen'
        }}
      />

      <Chat
        conversationId={activeConvId}
        conversationTitle={title}
        initialMessages={messages}
        loadingMsgs={loadingMsgs}
        onConversationCreated={handleConversationCreated}
        healthOk={healthOk}
        healthText={healthText}
        healthAt={healthAt}
        healthChecking={healthChecking}
        onRefreshHealth={refreshHealth}
        evidenceOpen={evidenceOpen}
        onSetEvidenceOpen={setEvidenceOpen}
        onOpenRail={() => setNavExpanded((v) => !v)}
      />

      {showDocPanel && (
        <>
          <div className="backdrop" onClick={() => setShowDocPanel(false)} />
          <DocumentPanel onClose={() => setShowDocPanel(false)} />
        </>
      )}
      {showFbPanel && (
        <>
          <div className="backdrop" onClick={() => setShowFbPanel(false)} />
          <FeedbackPanel onClose={() => setShowFbPanel(false)} />
        </>
      )}

      {/* 放最后：z-index 之上再叠一层，首次进入时盖住整个工作台 */}
      {showIntro && <Intro onClose={closeIntro} />}
    </div>
  )
}