import { useEffect, useRef, useState, type ReactNode } from 'react'
import {
  streamChat,
  submitFeedback,
  type Source,
  type TraceStep,
  type ChatMessageData,
} from '../api/chat'

interface Message {
  role: 'user' | 'assistant'
  content: string
  sources?: Source[]
  streaming?: boolean
  error?: boolean
  retryQuestion?: string
  messageId?: number
  rewrite?: string
  trace?: TraceStep[]
  feedbackSent?: boolean
}

const SUGGESTIONS = [
  '高血压患者能吃党参吗？',
  '小孩发烧39度怎么办？',
  '胃食管反流有哪些症状？',
  '糖尿病患者饮食注意事项？',
]

const CITE_RE = /\[(\d+)\]/g

/**
 * 把回答正文里的 [n] 渲染成可点击的引用锚点（点击滚到对应来源卡片并高亮）。
 *
 * 只对 sources 里**真实存在**的编号生效，其余原样留作纯文本。
 * 这一点是刻意的：无资料路径（insufficient / chat）的回答本就不该带编号，
 * 万一模型硬编了一个 [7]，宁可让它素着，也不能做成一个点不出东西的假链接——
 * 假链接比没有链接更糟，它会把"我编了个引用"伪装成"引用可溯源"。
 */
function renderAnswer(
  content: string,
  msgKey: string,
  sources: Source[] | undefined,
  onCite: (msgKey: string, n: number) => void,
): ReactNode {
  if (!sources || sources.length === 0 || !content.includes('[')) return content
  const known = new Set(sources.map((s) => s.index))
  const out: ReactNode[] = []
  let last = 0
  let hit = false
  CITE_RE.lastIndex = 0
  for (let m = CITE_RE.exec(content); m; m = CITE_RE.exec(content)) {
    const n = Number(m[1])
    if (!known.has(n)) continue
    hit = true
    if (m.index > last) out.push(content.slice(last, m.index))
    out.push(
      <button
        type="button"
        className="cite"
        key={`${msgKey}-c${m.index}`}
        title={`跳转到来源 [${n}]`}
        onClick={() => onCite(msgKey, n)}
      >
        [{n}]
      </button>,
    )
    last = m.index + m[0].length
  }
  if (!hit) return content
  if (last < content.length) out.push(content.slice(last))
  return out
}

export default function Chat({
  conversationId,
  initialMessages,
  onConversationCreated,
}: {
  conversationId: number | null
  initialMessages: ChatMessageData[]
  onConversationCreated: (id: number) => void
}) {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [flashId, setFlashId] = useState<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)
  const endRef = useRef<HTMLDivElement>(null)

  // 点正文里的 [n]：滚到对应来源卡片并闪一下。卡片 id 由「消息下标 + 编号」拼成，
  // 消息只追加不重排，所以下标稳定，不用额外维护引用表。
  function jumpToCite(msgKey: string, n: number) {
    const id = `cite-${msgKey}-${n}`
    const el = document.getElementById(id)
    if (!el) return
    el.scrollIntoView({ behavior: 'smooth', block: 'center' })
    setFlashId(id)
    window.setTimeout(() => setFlashId((cur) => (cur === id ? null : cur)), 1600)
  }

  // 切换会话时，加载历史消息（流式生成中跳过，避免清空正在显示的消息）
  useEffect(() => {
    if (loading) return
    if (initialMessages && initialMessages.length > 0) {
      setMessages(
        initialMessages.map((m) => ({
          role: m.role,
          content: m.content,
          sources: m.sources || undefined,
          messageId: m.id,
        })),
      )
    } else {
      setMessages([])
    }
  }, [conversationId, initialMessages])

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  async function send(question: string) {
    const q = question.trim()
    if (!q || loading) return

    setInput('')
    setLoading(true)

    const controller = new AbortController()
    abortRef.current = controller
    const timeoutId = setTimeout(() => controller.abort(), 120000)

    // 添加用户消息 + 占位的助手消息
    setMessages((m) => [
      ...m,
      { role: 'user', content: q },
      { role: 'assistant', content: '', streaming: true, retryQuestion: q },
    ])

    let gotSources = false
    let gotError = false

    await streamChat(
      q,
      conversationId,
      // onToken
      (text) => {
        setMessages((m) => {
          const copy = [...m]
          const last = copy[copy.length - 1]
          if (last && last.role === 'assistant') {
            copy[copy.length - 1] = { ...last, content: last.content + text }
          }
          return copy
        })
      },
      // onSources
      (sources) => {
        gotSources = true
        setMessages((m) => {
          const copy = [...m]
          const last = copy[copy.length - 1]
          if (last && last.role === 'assistant') {
            copy[copy.length - 1] = { ...last, sources, streaming: false }
          }
          return copy
        })
      },
      // onConversationId
      (id) => {
        onConversationCreated(id)
      },
      // onError
      (msg) => {
        gotError = true
        setMessages((m) => {
          const copy = [...m]
          const last = copy[copy.length - 1]
          if (last && last.role === 'assistant') {
            copy[copy.length - 1] = {
              ...last,
              content: last.content ? `${last.content}\n\n❌ ${msg}` : `❌ ${msg}`,
              streaming: false,
              error: true,
            }
          }
          return copy
        })
      },
      // onRewrite — 记录改写后的检索词
      (rewrite) => {
        setMessages((m) => {
          const copy = [...m]
          const last = copy[copy.length - 1]
          if (last && last.role === 'assistant') {
            copy[copy.length - 1] = { ...last, rewrite }
          }
          return copy
        })
      },
      // onTrace — 记录检索链路各步（后端推的是累计数组，直接整体替换）
      (steps) => {
        setMessages((m) => {
          const copy = [...m]
          const last = copy[copy.length - 1]
          if (last && last.role === 'assistant') {
            copy[copy.length - 1] = { ...last, trace: steps }
          }
          return copy
        })
      },
      // onStreamEnd — 检测流中断
      () => {
        if (!gotSources && !gotError) {
          setMessages((m) => {
            const copy = [...m]
            const last = copy[copy.length - 1]
            if (last && last.role === 'assistant' && last.streaming) {
              copy[copy.length - 1] = {
                ...last,
                content: last.content
                  ? `${last.content}\n\n⚠️ 响应流已中断，请点击重试。`
                  : '⚠️ 响应流已中断，未收到任何内容，请点击重试。',
                streaming: false,
                error: true,
              }
            }
            return copy
          })
        }
      },
      controller.signal,
    ).catch((e) => {
      if (e.name === 'AbortError') {
        setMessages((m) => {
          const copy = [...m]
          const last = copy[copy.length - 1]
          if (last && last.role === 'assistant' && last.streaming) {
            copy[copy.length - 1] = {
              ...last,
              content: last.content
                ? `${last.content}\n\n⚠️ 请求超时，请点击重试。`
                : '⚠️ 请求超时（2分钟无响应），请点击重试。',
              streaming: false,
              error: true,
            }
          }
          return copy
        })
      }
    }).finally(() => {
      clearTimeout(timeoutId)
      abortRef.current = null
    })

    setLoading(false)
  }

  function stopGeneration() {
    abortRef.current?.abort()
  }

  function retry(index: number) {
    const msg = messages[index]
    if (!msg?.retryQuestion) return
    // 移除失败的消息和对应的用户消息
    setMessages((m) => m.filter((_, i) => i !== index && i !== index - 1))
    // 重新发送
    setTimeout(() => send(msg.retryQuestion!), 0)
  }

  function handleFeedback(messageId: number, thumbs: 'up' | 'down') {
    submitFeedback(messageId, thumbs)
      .then(() => {
        setMessages((m) =>
          m.map((msg) =>
            msg.messageId === messageId ? { ...msg, feedbackSent: true } : msg,
          ),
        )
      })
      .catch((e) => console.error('反馈提交失败:', e))
  }

  return (
    <div className="chat">
      <div className="messages">
        {messages.length === 0 && (
          <div className="empty">
            <div className="empty-title">🏥 医疗智能问答助手</div>
            <div className="empty-desc">
              基于医疗问答数据集 + DeepSeek 大模型，回答仅供参考，不替代医师诊断。
            </div>
            <div className="suggestions">
              {SUGGESTIONS.map((s) => (
                <button
                  key={s}
                  className="suggestion"
                  onClick={() => send(s)}
                  disabled={loading}
                  style={loading ? { opacity: 0.5, cursor: 'not-allowed' } : undefined}
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((m, i) => (
          <div key={i} className={`bubble ${m.role}`}>
            <div className="bubble-role">
              {m.role === 'user' ? '🧑 你' : '🩺 医疗助手'}
            </div>
            <div className="bubble-content">
              {renderAnswer(m.content, String(i), m.sources, jumpToCite)}
              {m.streaming && <span className="cursor">▍</span>}
            </div>
            {m.error && m.retryQuestion && (
              <button className="retry-btn" onClick={() => retry(i)}>
                🔄 重试
              </button>
            )}
            {m.rewrite && (
              <details className="rewrite-note">
                <summary>🔍 实际检索词</summary>
                <div>{m.rewrite}</div>
              </details>
            )}
            {m.trace && m.trace.length > 0 && (
              <details className="trace-panel" open>
                <summary>🧭 检索过程</summary>
                <ol className="trace-list">
                  {m.trace.map((t, i) => (
                    <li key={i}>
                      <span className="trace-step">{t.step}</span>
                      <span className="trace-detail">
                        {Object.entries(t)
                          .filter(([k, v]) => k !== 'step' && v !== '' && v != null)
                          .map(([k, v]) => `${k}: ${String(v)}`)
                          .join('  ·  ')}
                      </span>
                    </li>
                  ))}
                </ol>
              </details>
            )}
            {m.sources && m.sources.length > 0 && (
              <div className="sources">
                <div className="sources-title">📎 引用来源 ({m.sources.length})</div>
                {m.sources.map((s) => (
                  <div
                    key={s.index}
                    id={`cite-${i}-${s.index}`}
                    className={
                      'source-card' +
                      (flashId === `cite-${i}-${s.index}` ? ' source-flash' : '')
                    }
                  >
                    <span className="source-index">[{s.index}]</span>
                    <span className="source-dept">{s.department || '未知科室'}</span>
                    <div className="source-title">{s.title}</div>
                    <div className="source-snippet">{s.snippet}</div>
                    {s.full_text && (
                      <details className="source-details">
                        <summary>查看完整原文</summary>
                        <div className="source-fulltext">{s.full_text}</div>
                      </details>
                    )}
                    <div className="source-file">来源: {s.source}</div>
                  </div>
                ))}
              </div>
            )}
            {m.role === 'assistant' &&
              m.messageId &&
              m.messageId > 0 &&
              (m.feedbackSent ? (
                <div className="feedback-done">✅ 已收到反馈，感谢你的帮助</div>
              ) : (
                <div className="feedback-bar">
                  <button onClick={() => handleFeedback(m.messageId!, 'up')}>
                    👍 有用
                  </button>
                  <button onClick={() => handleFeedback(m.messageId!, 'down')}>
                    👎 不准确
                  </button>
                </div>
              ))}
          </div>
        ))}
        <div ref={endRef} />
      </div>

      <form
        className="input-bar"
        onSubmit={(e) => {
          e.preventDefault()
          send(input)
        }}
      >
        <input
          type="text"
          value={input}
          placeholder="请描述你的症状或医疗问题..."
          onChange={(e) => setInput(e.target.value)}
          disabled={loading}
        />
        {loading ? (
          <button type="button" className="stop-btn" onClick={stopGeneration}>
            ⏹ 停止
          </button>
        ) : (
          <button type="submit" disabled={!input.trim()}>
            发送
          </button>
        )}
      </form>
    </div>
  )
}
