import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import CopyButton from './CopyButton'
import Icon from './Icon'
import EvidencePanel, {
  type EvidenceBlockKey,
  type EvidencePayload,
} from './EvidencePanel'
import CaseSummaryModal from './CaseSummaryModal'
import { formatDuration, formatFullStamp, formatStamp } from '../utils/time'
import {
  streamChat,
  submitFeedback,
  withdrawFeedback,
  type Source,
  type TraceStep,
  type ChatMessageData,
  type ClarificationRequest,
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
  /** 证据不足时后端中断追问（阶段二）：本条气泡是追问话术，下一条输入作为补充 */
  clarify?: ClarificationRequest
  /**
   * 本条消息的产生时间（ISO 串），用来显示"提问时间 / 回复时间"。
   *
   * 口径刻意分成两条来源：
   * - 用户消息用**本地发送时刻** —— "用户提问的时间"就是按下发送那一刻，
   *   这才是用户视角的语义；它和后端落库时间只差一次请求的网络耗时，
   *   在"到分钟"的显示粒度下不可见。
   * - 助手消息用**服务端落库时间**（由 message_id 事件带回）。不能用本地时间：
   *   刷新页面后时间改从数据库读，本地时钟与服务端有偏差就会让同一个回答的
   *   时间在刷新前后跳变 —— 同一个东西显示两个值是最容易让人怀疑数据的形态。
   */
  createdAt?: string
}

/** 👎 展开的纠错表单内容（按 messageId 定位） */
interface FeedbackForm {
  messageId: number
  corrected: string
  comment: string
}

const SUGGESTIONS = [
  '高血压患者能吃党参吗？',
  '小孩发烧39度怎么办？',
  '胃食管反流有哪些症状？',
  '糖尿病患者饮食注意事项？',
]

/** 追问选项预设库（按常见病程、伴随症状、部位与基础状况分类） */
const CLARIFY_PRESETS = [
  {
    category: '持续时间',
    chips: ['发病持续1-2天', '持续约1周', '反复发作超1个月'],
  },
  {
    category: '伴随症状',
    chips: ['伴发热/寒战', '伴恶心/呕吐', '伴剧烈疼痛', '无明显伴随症状'],
  },
  {
    category: '部位/既往',
    chips: ['部位在腹部/胃区', '部位在胸部/背部', '有高血压/慢病', '无基础病'],
  },
]

/** 急危重症与急救高危词线索 */
const EMERGENCY_CUES: readonly string[] = [
  '120', '拨打120', '立即前往急诊', '急救', '急诊',
  '胸痛', '呼吸困难', '喘不上气', '大出血', '意识丧失', '昏迷',
  '晕厥', '抽搐', '惊厥', '服毒', '自杀', '药物过量', '心肌梗死', '心梗',
]

function checkIsEmergency(content: string, userQuestion?: string): boolean {
  if (!content) return false
  const lowerContent = content.toLowerCase()
  if (lowerContent.includes('120') || lowerContent.includes('拨打120') || lowerContent.includes('急诊')) {
    return true
  }
  const combined = (content + ' ' + (userQuestion || '')).toLowerCase()
  return EMERGENCY_CUES.some((cue) => combined.includes(cue.toLowerCase()))
}

const CITE_RE = /\[(\d{1,2})\]/g
const BOLD_RE = /\*\*([^*]+)\*\*/g
const BULLET_RE = /^\s*(?:[-*•·]|\d{1,2}[.、)])\s+(.*)$/

/* ============================================================
   正文解析
   后端给的是纯文本（模型自由发挥 + 代码确定性追加的免责声明），
   没有结构化字段可用，所以这里做一次**保守**的轻解析：
   认得出列表/小标题/加粗就排版，认不出就原样当段落 —— 最差情况
   只是"看起来普通"，绝不会丢内容或把句子切碎。
   ============================================================ */
type Block =
  | { kind: 'p'; text: string }
  | { kind: 'h'; text: string }
  | { kind: 'ul'; items: string[] }

function isHeading(line: string): boolean {
  if (/^#{1,6}\s+/.test(line)) return true
  // 短句 + 冒号结尾 = 小节标题（如"可能的病因："）。
  // 带句号的整句排除掉，避免把正常句子认成标题。
  return line.length <= 20 && /[：:]$/.test(line) && !/[。！？]/.test(line)
}

function parseAnswer(content: string): Block[] {
  const lines = content.replace(/\r\n?/g, '\n').split('\n')
  const blocks: Block[] = []
  let para: string[] = []

  const flushPara = () => {
    const text = para.join('\n').trim()
    if (text) blocks.push({ kind: 'p', text })
    para = []
  }

  for (const raw of lines) {
    const line = raw.trim()
    if (!line) {
      flushPara()
      continue
    }
    const bullet = BULLET_RE.exec(line)
    if (bullet) {
      const last = blocks[blocks.length - 1]
      if (last && last.kind === 'ul') last.items.push(bullet[1])
      else {
        flushPara()
        blocks.push({ kind: 'ul', items: [bullet[1]] })
      }
      continue
    }
    if (isHeading(line)) {
      flushPara()
      blocks.push({ kind: 'h', text: line.replace(/^#{1,6}\s+/, '').replace(/[：:]$/, '') })
      continue
    }
    para.push(line)
  }
  flushPara()
  return blocks
}

/** 把 `**加粗**` 还原成 <strong>，避免模型输出的 Markdown 记号裸露在正文里 */
function renderPlain(text: string, keyBase: string): ReactNode[] {
  if (!text.includes('**')) return [text]
  const out: ReactNode[] = []
  let last = 0
  let m: RegExpExecArray | null
  BOLD_RE.lastIndex = 0
  while ((m = BOLD_RE.exec(text)) !== null) {
    if (m.index > last) out.push(text.slice(last, m.index))
    out.push(<strong key={`${keyBase}-b${m.index}`}>{m[1]}</strong>)
    last = m.index + m[0].length
  }
  if (last < text.length) out.push(text.slice(last))
  return out
}

/**
 * 把正文里的 [n] 渲染成可点击的引用角标（点击定位到右侧证据卡并高亮）。
 *
 * 只对 sources 里**真实存在**的编号生效，其余原样留作纯文本。
 * 这一点是刻意的：无资料路径（insufficient / chat）的回答本就不该带编号，
 * 万一模型硬编了一个 [7]，宁可让它素着，也不能做成一个点不出东西的假链接——
 * 假链接比没有链接更糟，它会把"我编了个引用"伪装成"引用可溯源"。
 */
function renderInline(
  text: string,
  msgIndex: number,
  known: Set<number>,
  onCite: (msgIndex: number, n: number) => void,
  flash: { i: number; n: number } | null,
): ReactNode {
  if (known.size === 0 || !text.includes('[')) return renderPlain(text, `m${msgIndex}`)
  const out: ReactNode[] = []
  const re = new RegExp(CITE_RE.source, 'g')
  let last = 0
  let hit = false
  let m: RegExpExecArray | null
  while ((m = re.exec(text)) !== null) {
    const n = Number(m[1])
    if (!known.has(n)) continue
    hit = true
    if (m.index > last) {
      out.push(...renderPlain(text.slice(last, m.index), `m${msgIndex}-${m.index}`))
    }
    const flashing = flash !== null && flash.i === msgIndex && flash.n === n
    out.push(
      <button
        key={`m${msgIndex}-c${m.index}`}
        type="button"
        className={`cite${flashing ? ' is-flash' : ''}`}
        title={`定位到证据 [${n}]`}
        onClick={() => onCite(msgIndex, n)}
      >
        [{n}]
      </button>,
    )
    last = m.index + m[0].length
  }
  if (!hit) return renderPlain(text, `m${msgIndex}`)
  if (last < text.length) {
    out.push(...renderPlain(text.slice(last), `m${msgIndex}-tail`))
  }
  return out
}

/** 报告式正文：结论先行 → 小节（带色标）→ 条目；⚠️ 开头的段落收成提示条 */
function renderAnswer(
  content: string,
  msgIndex: number,
  sources: Source[] | undefined,
  onCite: (msgIndex: number, n: number) => void,
  flash: { i: number; n: number } | null,
): ReactNode {
  const known = new Set((sources || []).map((s) => s.index))
  const inline = (t: string) => renderInline(t, msgIndex, known, onCite, flash)

  const blocks = parseAnswer(content)
  let lead: string | null = null
  const rest: Block[] = []
  for (const b of blocks) {
    if (lead === null && b.kind === 'p') {
      lead = b.text
      continue
    }
    rest.push(b)
  }

  const groups: { title?: string; items: Block[] }[] = []
  for (const b of rest) {
    if (b.kind === 'h') groups.push({ title: b.text, items: [] })
    else {
      if (groups.length === 0) groups.push({ items: [] })
      groups[groups.length - 1].items.push(b)
    }
  }

  return (
    <div className="answer">
      {lead !== null && <p className="answer-lead">{inline(lead)}</p>}
      {groups.map((g, gi) => (
        <div className="answer-section" key={gi}>
          {g.title && <div className="answer-section-title">{g.title}</div>}
          {g.items.map((b, bi) => {
            if (b.kind === 'ul') {
              return (
                <ul className="answer-list" key={bi}>
                  {b.items.map((it, ii) => (
                    <li key={ii}>{inline(it)}</li>
                  ))}
                </ul>
              )
            }
            if (b.kind !== 'p') return null
            // 免责声明/忠实性提示由后端代码确定性追加，固定以 ⚠️ 开头，
            // 单独收成提示条比混在正文里更清楚它是"声明"而不是"结论"。
            if (b.text.startsWith('⚠️')) {
              return (
                <div className="answer-note" key={bi}>
                  <Icon name="alert" size="sm" />
                  <span>{b.text.replace(/^⚠️\s*/, '')}</span>
                </div>
              )
            }
            return <p key={bi}>{inline(b.text)}</p>
          })}
        </div>
      ))}
    </div>
  )
}

export default function Chat({
  conversationId,
  conversationTitle,
  initialMessages,
  loadingMsgs,
  onConversationCreated,
  healthOk,
  healthText,
  healthAt,
  healthChecking,
  onRefreshHealth,
  evidenceOpen,
  onSetEvidenceOpen,
  onOpenRail,
}: {
  conversationId: number | null
  conversationTitle: string
  initialMessages: ChatMessageData[]
  loadingMsgs: boolean
  onConversationCreated: (id: number) => void
  healthOk: boolean
  healthText: string
  healthAt: string
  healthChecking: boolean
  onRefreshHealth: () => void
  evidenceOpen: boolean
  onSetEvidenceOpen: (open: boolean) => void
  onOpenRail: () => void
}) {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [activeIdx, setActiveIdx] = useState<number | null>(null)
  const [activeCite, setActiveCite] = useState<number | null>(null)
  const [flash, setFlash] = useState<{ i: number; n: number } | null>(null)
  const [openBlocks, setOpenBlocks] = useState<Record<EvidenceBlockKey, boolean>>({
    rewrite: true,
    trace: true,
    sources: true,
  })
  const [fbForm, setFbForm] = useState<FeedbackForm | null>(null)
  // 错误挂在具体那条消息上：反馈按钮是每条消息各自的，报错也得让用户知道是哪条的失败
  const [fbErr, setFbErr] = useState<{ messageId: number; text: string } | null>(null)
  // 交互式追问选项胶囊所选内容
  const [selectedClarifyChips, setSelectedClarifyChips] = useState<string[]>([])
  // 急症自救指引展开状态 (按消息下标记录)
  const [openGuideMap, setOpenGuideMap] = useState<Record<number, boolean>>({})
  // 门诊预问诊小结抽屉状态
  const [showSummaryModal, setShowSummaryModal] = useState(false)
  const abortRef = useRef<AbortController | null>(null)
  const endRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  function toggleClarifyChip(chip: string) {
    setSelectedClarifyChips((prev) =>
      prev.includes(chip) ? prev.filter((c) => c !== chip) : [...prev, chip],
    )
  }

  function sendClarifyChips(chips: string[]) {
    if (chips.length === 0) return
    const text = chips.join('，')
    setSelectedClarifyChips([])
    send(text)
  }

  function appendChipsToInput(chips: string[]) {
    if (chips.length === 0) return
    const text = chips.join('，')
    setInput((prev) => (prev.trim() ? `${prev.trim()}，${text}` : text))
    setSelectedClarifyChips([])
    inputRef.current?.focus()
  }
  /**
   * 消息数组的实时镜像。
   * 为什么需要：send() 里要算出"这一轮助手消息会落在哪个下标"，
   * 好在来源/追问事件到达时把证据栏切过去；而闭包里的 messages 是旧的。
   * 用 setMessages 的更新函数里再 setState 来偷这个下标是不行的 ——
   * 更新函数必须是纯函数，StrictMode 下会被调用两次。
   */
  const messagesRef = useRef<Message[]>([])
  useEffect(() => {
    messagesRef.current = messages
  }, [messages])

  /**
   * 点正文角标 / 点「证据栏查看」的统一入口。
   * citeN 有值时才闪烁角标并让证据栏滚到对应卡片；证据栏关着就先打开，
   * 否则"点了没反应"。
   */
  function focusMessage(i: number, citeN: number | null) {
    setActiveIdx(i)
    setActiveCite(citeN)
    if (citeN != null) {
      setFlash({ i, n: citeN })
      window.setTimeout(() => setFlash((cur) => (cur && cur.i === i && cur.n === citeN ? null : cur)), 1900)
    }
    if (!evidenceOpen) onSetEvidenceOpen(true)
  }

  // 切换会话时加载历史消息（流式生成中跳过，避免清空正在显示的消息）
  useEffect(() => {
    if (loading) return
    if (initialMessages && initialMessages.length > 0) {
      const mapped: Message[] = initialMessages.map((m) => ({
        role: m.role,
        content: m.content,
        sources: m.sources || undefined,
        messageId: m.id,
        // 历史消息的时间直接来自库（后端 /messages 已返回 created_at），
        // 刷新后与刚生成时显示的必须是同一个值
        createdAt: m.created_at || undefined,
        // 后端消息里带回了该条是否已反馈（幂等），刷新后按钮状态不丢
        feedbackSent: m.feedback?.thumbs === 'up' || m.feedback?.thumbs === 'down',
        // 追问话术：后端带回了 is_clarification，据此重建 clarify 对象，
        // 刷新后「请补充信息」徽章才不会退化成普通回答。
        clarify: m.is_clarification
          ? { type: 'clarification_request' as const, message: m.content }
          : undefined,
      }))
      setMessages(mapped)
      // 默认把证据栏绑到最后一条有来源的回答上（用户最可能想看的那轮）
      let last = -1
      mapped.forEach((m, i) => {
        if (m.role === 'assistant' && (m.sources || []).length > 0) last = i
      })
      setActiveIdx(last >= 0 ? last : null)
      setActiveCite(null)
    } else {
      setMessages([])
      setActiveIdx(null)
      setActiveCite(null)
    }
  }, [conversationId, initialMessages])

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  async function send(question: string) {
    const q = question.trim()
    if (!q || loading) return

    setInput('')
    if (inputRef.current) inputRef.current.style.height = 'auto'
    setLoading(true)

    const controller = new AbortController()
    abortRef.current = controller
    const timeoutId = setTimeout(() => controller.abort(), 120000)

    // 添加用户消息 + 占位的助手消息。
    // 用户消息当场盖上本地发送时间；助手消息的时间要等服务端落库后回传
    // （见 onMessageId），此刻留空 —— 先用本地时间打底会与刷新后的值不一致。
    const assistantIdx = messagesRef.current.length + 1
    setMessages((m) => [
      ...m,
      { role: 'user', content: q, createdAt: new Date().toISOString() },
      { role: 'assistant', content: '', streaming: true, retryQuestion: q },
    ])
    setActiveIdx(null)
    setActiveCite(null)

    let gotSources = false
    let gotError = false
    let gotClarify = false

    /** 改写最后一条助手消息。流式期间每条 token 都会走这里，所以必须只做最小拷贝。 */
    const patchLast = (patch: Partial<Message>) => {
      setMessages((m) => {
        const copy = [...m]
        const last = copy[copy.length - 1]
        if (last && last.role === 'assistant') copy[copy.length - 1] = { ...last, ...patch }
        return copy
      })
    }

    /**
     * 只在还没有时间时补一个本地时间。
     * 报错 / 超时 / 客户端中断这几条路，后端根本不会回传 created_at（那条消息
     * 没落库），留空会和相邻气泡不一致；但服务端已经给过值时**不能覆盖**，
     * 否则又变成"刷新前后两个值"。
     */
    const patchTimeIfEmpty = () => {
      setMessages((m) => {
        const copy = [...m]
        const last = copy[copy.length - 1]
        if (last && last.role === 'assistant' && !last.createdAt) {
          copy[copy.length - 1] = { ...last, createdAt: new Date().toISOString() }
        }
        return copy
      })
    }

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
      // onSources —— 来源到达即视为"这一轮有证据"，顺手把证据栏切到这条回答
      (sources) => {
        gotSources = true
        patchLast({ sources, streaming: false })
        setActiveIdx(assistantIdx)
        setActiveCite(null)
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
      (rewrite) => patchLast({ rewrite }),
      // onClarification —— 证据不足，后端中断追问（阶段二）。
      // content 直接用追问话术：这轮没有 token 流，气泡本体就是这句话。
      (c) => {
        gotClarify = true
        patchLast({ content: c.message, clarify: c, streaming: false })
        setActiveIdx(assistantIdx)
        setActiveCite(null)
      },
      // onTrace — 记录检索链路各步（后端推的是累计数组，直接整体替换）
      (steps) => patchLast({ trace: steps }),
      // onMessageId — 后端落库后回传的 {id, created_at}。
      // id 没有它 `m.messageId` 恒为 undefined，👍/👎 就永远不渲染；
      // created_at 没有它，"系统回复时间"只能退回本地时钟，刷新后会跳变。
      (meta) =>
        patchLast(
          meta.created_at
            ? { messageId: meta.id, createdAt: meta.created_at }
            : { messageId: meta.id },
        ),
      // onStreamEnd — 检测流中断。
      // gotClarify 必须一并排除：澄清流恰好就是「没有 sources 就结束」的流，
      // 漏了它每次追问都会被误报成"响应流已中断"。
      () => {
        if (!gotSources && !gotError && !gotClarify) {
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
      // onCorrection —— 忠实性校验修正（阶段四）：整段替换已显示的回答。
      // token 事件只能追加，所以这是前端**唯一**能反映"剥除越界引用编号"的入口。
      (c) => patchLast({ content: c.answer }),
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
      // 兜底补时间：正常路径上 onMessageId 已经给过服务端时间，这里不会生效
      patchTimeIfEmpty()
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

  function markFeedbackDone(messageId: number) {
    setMessages((m) =>
      m.map((msg) => (msg.messageId === messageId ? { ...msg, feedbackSent: true } : msg)),
    )
  }

  // 👍 直接提交；👎 不直接提交，先展开纠错表单，让用户可选填「正确回答/备注」——
  // corrected_answer 是看板最看重的标注数据，但要有采集入口它才可能被填进来。
  function openDownForm(messageId: number) {
    setFbErr(null)
    setFbForm({ messageId, corrected: '', comment: '' })
  }

  function submitUp(messageId: number) {
    setFbErr(null)
    submitFeedback(messageId, 'up')
      .then(() => markFeedbackDone(messageId))
      .catch((e) => setFbErr({ messageId, text: `反馈提交失败：${e.message}` }))
  }

  function submitDownForm() {
    if (!fbForm) return
    const { messageId } = fbForm
    setFbErr(null)
    submitFeedback(messageId, 'down', fbForm.corrected, fbForm.comment)
      .then(() => {
        markFeedbackDone(messageId)
        setFbForm(null)
      })
      .catch((e) => setFbErr({ messageId, text: `反馈提交失败：${e.message}` }))
  }

  // 点错反悔：撤回已提交的反馈，按钮重新出现
  function withdraw(messageId: number) {
    setFbErr(null)
    withdrawFeedback(messageId)
      .then(() => {
        setMessages((m) =>
          m.map((msg) => (msg.messageId === messageId ? { ...msg, feedbackSent: false } : msg)),
        )
      })
      .catch((e) => setFbErr({ messageId, text: `撤回反馈失败：${e.message}` }))
  }

  // 证据栏要的那一轮材料。只依赖 activeIdx + messages，不额外存一份副本，
  // 免得流式更新时两处状态不同步。
  const evidencePayload = useMemo<EvidencePayload | null>(() => {
    if (activeIdx === null) return null
    const m = messages[activeIdx]
    if (!m || m.role !== 'assistant') return null
    const sources = m.sources || []
    let round = 0
    for (let i = 0; i <= activeIdx; i++) if (messages[i].role === 'assistant') round++
    return {
      msgIndex: activeIdx,
      scope: `关联第 ${round} 轮回答 · 点击角标可定位`,
      sources,
      rewrite: m.rewrite,
      trace: m.trace,
      streaming: !!m.streaming,
      noEvidence: !m.streaming && sources.length === 0,
    }
  }, [activeIdx, messages])

  return (
    <>
      <main className="workspace">
        <header className="topbar">
          <div className="topbar-title">
            <h1 title={conversationTitle}>{conversationTitle}</h1>
            <div className="topbar-sub">
              <span className={`dot${healthOk ? '' : ' is-off'}`} />
              <span className="status-text">{healthText}</span>
              {healthAt && (
                <>
                  <span className="chip-sep" />
                  <span>更新于 {healthAt}</span>
                </>
              )}
            </div>
          </div>
          <div className="topbar-actions">
            <button
              className="btn only-narrow"
              type="button"
              aria-label="展开会话列表"
              onClick={onOpenRail}
            >
              <Icon name="chat" size="sm" />
            </button>
            <button
              className="btn btn-icon"
              type="button"
              aria-label="刷新检索服务状态"
              title="刷新检索服务状态"
              disabled={healthChecking}
              onClick={onRefreshHealth}
            >
              <Icon name="refresh" size="sm" className={healthChecking ? 'spin' : ''} />
            </button>
            {messages.length > 0 && (
              <button
                className="btn"
                type="button"
                title="生成并导出本轮门诊就医病历小结"
                onClick={() => setShowSummaryModal(true)}
              >
                <Icon name="file" size="sm" />
                <span>问诊小结</span>
              </button>
            )}
            <button
              className={`btn${evidenceOpen ? ' is-on' : ''}`}
              type="button"
              aria-pressed={evidenceOpen}
              onClick={() => onSetEvidenceOpen(!evidenceOpen)}
            >
              <Icon name="panel" size="sm" />
              <span>证据栏</span>
            </button>
          </div>
        </header>

        <div className="disclaimer">
          <Icon name="alert" size="sm" />
          <span>
            本系统回答仅基于公开医学问答数据集与大模型生成，仅供参考，
            <b>不能替代执业医师诊断</b>。涉及急症请立即拨打 120。
          </span>
        </div>

        <div className="thread">
          <div className="thread-inner">
            {loadingMsgs && messages.length === 0 && (
              <div className="empty">
                <span className="empty-mark">
                  <Icon name="pulse" size="lg" className="spin" />
                </span>
                <p>正在载入这段对话…</p>
              </div>
            )}

            {!loadingMsgs && messages.length === 0 && (
              <div className="empty">
                <span className="empty-mark">
                  <Icon name="pulse" size="lg" />
                </span>
                <h2>医疗智能问答助手</h2>
                <p>
                  基于医疗问答数据集与大模型生成，每条回答都标注来源，可在右侧逐条核对原文。
                  回答仅供参考，不替代医师诊断。
                </p>
                <div className="suggestions">
                  {SUGGESTIONS.map((s) => (
                    <button
                      key={s}
                      className="suggestion"
                      type="button"
                      onClick={() => send(s)}
                      disabled={loading}
                    >
                      {s}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {messages.map((m, i) => {
              const isActive = activeIdx === i
              const cls = [
                'msg',
                m.role,
                isActive ? 'is-active' : '',
                m.clarify ? 'is-clarify' : '',
                m.error ? 'is-error' : '',
              ]
                .filter(Boolean)
                .join(' ')
              // 助手消息的"耗时"= 本条回复时间 − 紧邻上一条用户消息的提问时间。
              // 消息严格 user/assistant 交替，所以 i-1 就是它回答的那个问题。
              const prevUserCreatedAt =
                i > 0 && messages[i - 1]?.role === 'user' ? messages[i - 1].createdAt : undefined
              const elapsed =
                m.role === 'assistant' ? formatDuration(prevUserCreatedAt, m.createdAt) : ''
              const stamp = formatStamp(m.createdAt)
              return (
                <article key={i} className={cls}>
                  <div className="msg-role">
                    <span className="who">
                      <Icon name={m.role === 'user' ? 'user' : 'stethoscope'} size="sm" />
                      {m.role === 'user' ? '你' : '医疗助手'}
                    </span>
                    {/* 提问时间 / 回复时间。带上「提问」「回复」前缀而不是裸时:分 ——
                        两条气泡的时间经常落在同一分钟，光看数字分不出哪个是哪个。
                        悬浮给完整时间戳（含秒），对齐后端日志时不用去翻库。
                        流式期间助手消息还没有服务端时间，此处自然为空，不占位跳动。 */}
                    {stamp && (
                      <span
                        className="msg-time"
                        title={
                          m.role === 'user'
                            ? `提问时间：${formatFullStamp(m.createdAt)}`
                            : `回复时间：${formatFullStamp(m.createdAt)}`
                        }
                      >
                        {m.role === 'user' ? '提问' : '回复'} {stamp}
                        {elapsed && <span className="msg-elapsed">· 耗时 {elapsed}</span>}
                      </span>
                    )}
                    {m.clarify && (
                      <span className="clarify-badge">
                        <Icon name="search" size="sm" />
                        请补充信息
                      </span>
                    )}
                  </div>

                  <div className="msg-body">
                    {m.role === 'user' ? (
                      m.content
                    ) : m.streaming ? (
                      <div className="answer">
                        <p className="answer-lead answer-stream">
                          {m.content}
                          <span className="cursor" />
                        </p>
                      </div>
                    ) : (
                      <>
                        {(() => {
                          const prevUserMsg =
                            i > 0 && messages[i - 1]?.role === 'user'
                              ? messages[i - 1]?.content
                              : m.retryQuestion
                          const isEmergency = checkIsEmergency(m.content, prevUserMsg)
                          if (!isEmergency) return null
                          return (
                            <aside className="emergency-triage-card" aria-label="急症紧急就医提醒">
                              <div className="emergency-triage-head">
                                <div className="emergency-triage-title">
                                  <span className="emergency-pulse-dot" />
                                  <Icon name="alert" size="sm" />
                                  <span>危急重症预警 / 紧急就医提醒</span>
                                </div>
                              </div>
                              <div className="emergency-triage-body">
                                检测到您咨询的情况可能涉及急性重症或突发险情。如伴有剧烈胸痛、呼吸急促困难、神志不清抽搐或大出血等，<strong>切勿延误时间，请立即采取应急措施就医！</strong>
                              </div>
                              <div className="emergency-triage-actions">
                                <a
                                  href="tel:120"
                                  className="btn-emergency-call"
                                  title="点击直接拨打120急救电话"
                                >
                                  <Icon name="pulse" size="sm" />
                                  一键呼叫 120 急救
                                </a>
                                <button
                                  type="button"
                                  className="emergency-guide-toggle"
                                  onClick={() =>
                                    setOpenGuideMap((prev) => ({ ...prev, [i]: !prev[i] }))
                                  }
                                >
                                  {openGuideMap[i] ? '收起急救指引 ▲' : '急诊就医与自救指引 ▼'}
                                </button>
                              </div>
                              {openGuideMap[i] && (
                                <div className="emergency-triage-guide">
                                  <strong>现场紧急应对与自救原则：</strong>
                                  <ul>
                                    <li>
                                      <strong>保持呼吸道通畅</strong>：解开患者衣领，若有呕吐偏头防误吸窒息，切勿强行搬动。
                                    </li>
                                    <li>
                                      <strong>就地平卧静息</strong>：胸痛、心慌发作时严禁盲目走动或做剧烈运动，保持安静坐卧。
                                    </li>
                                    <li>
                                      <strong>切勿盲目喂药</strong>：在未经急诊执业医师确诊前，避免盲目喂水、喂止痛药或降压药。
                                    </li>
                                    <li>
                                      <strong>备齐急救资料</strong>：备好患者身份证、医保卡及正在服用的药物包装盒，方便急救医护交接。
                                    </li>
                                  </ul>
                                </div>
                              )}
                            </aside>
                          )
                        })()}
                        {renderAnswer(m.content, i, m.sources, focusMessage, flash)}
                      </>
                    )}

                    {m.clarify && (
                      <div className="clarify-chips-container">
                        {i === messages.length - 1 && !loading ? (
                          <div className="clarify-chips-panel">
                            <div className="clarify-chips-title">
                              <Icon name="search" size="sm" />
                              <span>点击快捷填选补充信息（可多选组合）：</span>
                            </div>
                            {CLARIFY_PRESETS.map((group) => (
                              <div key={group.category} className="clarify-chips-group">
                                <span className="clarify-group-label">{group.category}：</span>
                                {group.chips.map((chip) => {
                                  const isSel = selectedClarifyChips.includes(chip)
                                  return (
                                    <button
                                      key={chip}
                                      type="button"
                                      className={`clarify-chip-btn${isSel ? ' is-selected' : ''}`}
                                      onClick={() => toggleClarifyChip(chip)}
                                    >
                                      {isSel && <Icon name="check" size="sm" />}
                                      {chip}
                                    </button>
                                  )
                                })}
                              </div>
                            ))}
                            <div className="clarify-chips-actions">
                              {selectedClarifyChips.length > 0 && (
                                <>
                                  <button
                                    type="button"
                                    className="btn btn-accent btn-sm"
                                    onClick={() => sendClarifyChips(selectedClarifyChips)}
                                  >
                                    <Icon name="send" size="sm" />
                                    发送所选（{selectedClarifyChips.length} 项）
                                  </button>
                                  <button
                                    type="button"
                                    className="btn btn-sm"
                                    onClick={() => appendChipsToInput(selectedClarifyChips)}
                                  >
                                    填入输入框
                                  </button>
                                </>
                              )}
                              <button
                                type="button"
                                className="btn btn-sm"
                                onClick={() => {
                                  setSelectedClarifyChips([])
                                  send('算了，先不用了')
                                }}
                              >
                                算了，不补充
                              </button>
                            </div>
                          </div>
                        ) : (
                          <div className="clarify-chips-done">
                            <Icon name="check" size="sm" />
                            已完成该轮补充
                          </div>
                        )}
                      </div>
                    )}
                  </div>

                  {/* 复制按钮紧跟正文：客户读完就想复制，放底部会被来源/反馈区隔开。
                      流式生成中不显示（内容还在变，复制到半截没有意义）。 */}
                  {!m.streaming && m.content && (
                    <div className="msg-tools">
                      <CopyButton
                        text={m.content}
                        label={m.role === 'user' ? '复制提问' : '复制回答'}
                        className="copy-mini"
                      />
                      {m.role === 'assistant' && (
                        <>
                          <span className="chip-sep" />
                          <button
                            className={`chip${isActive ? ' is-on' : ''}`}
                            type="button"
                            onClick={() => focusMessage(i, null)}
                          >
                            <Icon name="link" size="sm" />
                            证据栏查看
                            {(m.sources?.length ?? 0) > 0 && (
                              <span className="chip-num">{m.sources!.length}</span>
                            )}
                          </button>
                          {m.rewrite && (
                            <button
                              className="chip"
                              type="button"
                              onClick={() => {
                                setOpenBlocks((b) => ({ ...b, rewrite: true }))
                                focusMessage(i, null)
                              }}
                            >
                              <Icon name="search" size="sm" />
                              实际检索词
                            </button>
                          )}
                          {m.trace && m.trace.length > 0 && (
                            <button
                              className="chip"
                              type="button"
                              onClick={() => {
                                setOpenBlocks((b) => ({ ...b, trace: true }))
                                focusMessage(i, null)
                              }}
                            >
                              <Icon name="route" size="sm" />
                              检索过程
                            </button>
                          )}
                        </>
                      )}
                      {m.error && m.retryQuestion && (
                        <button className="chip chip-danger" type="button" onClick={() => retry(i)}>
                          <Icon name="refresh" size="sm" />
                          重试
                        </button>
                      )}
                    </div>
                  )}

                  {m.role === 'assistant' &&
                    m.messageId &&
                    m.messageId > 0 &&
                    !m.streaming && (
                      <>
                        {m.feedbackSent ? (
                          <div className="fb-done">
                            <Icon name="check" size="sm" />
                            已收到反馈，感谢你的帮助
                            <button type="button" onClick={() => withdraw(m.messageId!)}>
                              撤回
                            </button>
                          </div>
                        ) : (
                          <>
                            <div className="fb-bar">
                              <button className="chip" type="button" onClick={() => submitUp(m.messageId!)}>
                                <Icon name="up" size="sm" />
                                有用
                              </button>
                              <button
                                className="chip"
                                type="button"
                                onClick={() => openDownForm(m.messageId!)}
                                disabled={fbForm?.messageId === m.messageId}
                              >
                                <Icon name="down" size="sm" />
                                不准确
                              </button>
                            </div>
                            {fbForm && fbForm.messageId === m.messageId && (
                              <div className="fb-form">
                                <label htmlFor={`corr-${m.messageId}`}>
                                  如果能给出你认为更准确的回答，将帮助改进模型
                                </label>
                                <textarea
                                  id={`corr-${m.messageId}`}
                                  rows={3}
                                  autoFocus
                                  placeholder="（可选）更准确的回答"
                                  value={fbForm.corrected}
                                  onChange={(e) => setFbForm({ ...fbForm, corrected: e.target.value })}
                                />
                                <label htmlFor={`cmt-${m.messageId}`}>备注</label>
                                <textarea
                                  id={`cmt-${m.messageId}`}
                                  rows={2}
                                  placeholder="（可选）备注"
                                  value={fbForm.comment}
                                  onChange={(e) => setFbForm({ ...fbForm, comment: e.target.value })}
                                />
                                <div className="fb-form-actions">
                                  <button className="btn" type="button" onClick={() => setFbForm(null)}>
                                    取消
                                  </button>
                                  <button className="btn btn-accent" type="button" onClick={submitDownForm}>
                                    提交差评
                                  </button>
                                </div>
                              </div>
                            )}
                          </>
                        )}
                        {fbErr && fbErr.messageId === m.messageId && (
                          <div className="fb-err">
                            <Icon name="alert" size="sm" />
                            {fbErr.text}
                          </div>
                        )}
                      </>
                    )}
                </article>
              )
            })}
            <div ref={endRef} />
          </div>
        </div>

        <form
          className="composer"
          onSubmit={(e) => {
            e.preventDefault()
            send(input)
          }}
        >
          <div className="composer-inner">
            <label className="sr-only" htmlFor="chat-input">
              请输入你的医疗问题
            </label>
            <textarea
              id="chat-input"
              ref={inputRef}
              rows={1}
              value={input}
              placeholder="请描述你的症状或医疗问题…"
              onChange={(e) => setInput(e.target.value)}
              onInput={(e) => {
                const el = e.currentTarget
                el.style.height = 'auto'
                el.style.height = `${Math.min(el.scrollHeight, 160)}px`
              }}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                  e.preventDefault()
                  send(input)
                }
              }}
            />
            {loading ? (
              <button className="btn" type="button" onClick={stopGeneration}>
                <Icon name="stop" size="sm" />
                停止
              </button>
            ) : (
              <button className="btn btn-accent" type="submit" disabled={!input.trim()}>
                <Icon name="send" size="sm" />
                发送
              </button>
            )}
          </div>
          <p className="composer-hint">证据不足时系统会先向你追问，不会硬答。</p>
        </form>
      </main>

      <EvidencePanel
        payload={evidencePayload}
        activeCite={activeCite}
        openBlocks={openBlocks}
        onToggleBlock={(k) => {
          setOpenBlocks((b) => ({ ...b, [k]: !b[k] }))
          // 清掉角标定位：否则每次折叠都会被强制滚回那张证据卡
          setActiveCite(null)
        }}
        onSelectCite={(n) => {
          if (activeIdx !== null) focusMessage(activeIdx, n)
        }}
        onClose={() => onSetEvidenceOpen(false)}
      />

      {showSummaryModal && (
        <>
          <div className="backdrop" onClick={() => setShowSummaryModal(false)} />
          <CaseSummaryModal
            conversationTitle={conversationTitle}
            messages={messages}
            onClose={() => setShowSummaryModal(false)}
          />
        </>
      )}
    </>
  )
}