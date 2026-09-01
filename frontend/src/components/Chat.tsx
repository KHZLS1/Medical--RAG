import { useEffect, useRef, useState } from 'react'
import { streamChat, type Source } from '../api/chat'

interface Message {
  role: 'user' | 'assistant'
  content: string
  sources?: Source[]
  streaming?: boolean
}

const SUGGESTIONS = [
  '高血压患者能吃党参吗？',
  '小孩发烧39度怎么办？',
  '胃食管反流有哪些症状？',
  '糖尿病患者饮食注意事项？',
]

export default function Chat() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const endRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  async function send(question: string) {
    const q = question.trim()
    if (!q || loading) return

    setInput('')
    setLoading(true)

    // 添加用户消息 + 占位的助手消息
    setMessages((m) => [
      ...m,
      { role: 'user', content: q },
      { role: 'assistant', content: '', streaming: true },
    ])

    await streamChat(
      q,
      // onToken: 增量更新最后一条助手消息
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
      // onSources: 把来源挂到最后一条助手消息
      (sources) => {
        setMessages((m) => {
          const copy = [...m]
          const last = copy[copy.length - 1]
          if (last && last.role === 'assistant') {
            copy[copy.length - 1] = { ...last, sources, streaming: false }
          }
          return copy
        })
      },
      // onError
      (msg) => {
        setMessages((m) => {
          const copy = [...m]
          const last = copy[copy.length - 1]
          if (last && last.role === 'assistant') {
            copy[copy.length - 1] = {
              ...last,
              content: last.content + `\n\n❌ ${msg}`,
              streaming: false,
            }
          }
          return copy
        })
      },
    )

    setLoading(false)
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
                <button key={s} className="suggestion" onClick={() => send(s)}>
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
              {m.content}
              {m.streaming && <span className="cursor">▍</span>}
            </div>
            {m.sources && m.sources.length > 0 && (
              <div className="sources">
                <div className="sources-title">📎 引用来源 ({m.sources.length})</div>
                {m.sources.map((s) => (
                  <div key={s.index} className="source-card">
                    <span className="source-index">[{s.index}]</span>
                    <span className="source-dept">{s.department || '未知科室'}</span>
                    <div className="source-title">{s.title}</div>
                    <div className="source-snippet">{s.snippet}</div>
                    <div className="source-file">来源: {s.source}</div>
                  </div>
                ))}
              </div>
            )}
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
        <button type="submit" disabled={loading || !input.trim()}>
          {loading ? '生成中...' : '发送'}
        </button>
      </form>
    </div>
  )
}
