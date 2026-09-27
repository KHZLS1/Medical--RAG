// SSE 流式问答客户端 + 会话管理 API
// 后端 /api/chat 用 POST + SSE 返回，前端用 fetch + ReadableStream 解析

export interface Source {
  index: number
  department: string
  title: string
  source: string
  snippet: string
  full_text?: string
}

/** 检索链路单步记录（后端 trace 事件推送，字段随节点不同而变） */
export interface TraceStep {
  step: string
  [key: string]: unknown   // query / recalled / top_k / source / confidence / reason ...
}

/** 证据不足时后端中断追问（阶段二）。后端 interrupt() 的 value 原样透传。 */
export interface ClarificationRequest {
  type: 'clarification_request'
  message: string
  top_score?: number
}

/** 忠实性校验对回答的修正（阶段四）：剥除越界引用编号 / 追加提示后的完整文本 */
export interface CorrectionPayload {
  answer: string
  invalid_citations: number[]
  verdict: string
}

export interface Conversation {
  id: number
  title: string
  created_at: string
  updated_at: string
}

export interface ChatMessageData {
  id: number
  conversation_id: number
  role: 'user' | 'assistant'
  content: string
  sources?: Source[] | null
  created_at: string
  /** 本条消息是否已有反馈（后端返回 {id, thumbs}），assistant 消息可为 null */
  feedback?: { id: number; thumbs: 'up' | 'down' } | null
  /** 本条是否为证据不足的追问话术（阶段二）：刷新后据此恢复「🔎 请补充信息」徽章 */
  is_clarification?: boolean
}

/** 助手消息落库后回传的元信息（message_id 事件） */
export interface PersistedMessageMeta {
  id: number
  /** 落库时间（服务端）。显示"系统回复时间"以它为准，见后端 main.py 的说明 */
  created_at: string | null
}

interface SSEEvent {
  type:
    | 'token'
    | 'sources'
    | 'error'
    | 'conversation_id'
    | 'message_id'
    | 'rewrite'
    | 'trace'
    | 'clarification_request'
    | 'correction'
  data: string | Source[] | number | TraceStep[] | ClarificationRequest | CorrectionPayload | PersistedMessageMeta
}

/**
 * 流式问答（支持多轮上下文）
 * @param question 用户问题
 * @param conversationId 会话ID（可选，不传则后端自动新建会话）
 * @param onToken  每收到一段文本时的回调
 * @param onSources 收到引用来源时的回调
 * @param onConversationId 收到会话ID时的回调（新建会话时触发）
 * @param onError  出错回调
 * @param onRewrite 收到改写后检索词时的回调（可选）
 * @param onClarification 证据不足时后端中断追问的回调（可选）。
 *        收到它即本轮结束（无 token / sources），用户下一条输入会作为补充恢复对话
 * @param onTrace  收到检索链路各步（累计数组）时的回调（可选）
 * @param onMessageId 收到本条助手消息落库元信息时的回调（可选）
 *        —— 没有它，刚生成完的回答拿不到 message_id，👍/👎 按钮就不渲染；
 *        也拿不到服务端落库时间，"系统回复时间"就只能用客户端本地时钟，
 *        刷新后同一个回答的时间会跳变
 * @param onStreamEnd 流结束回调（可选）
 * @param onCorrection 忠实性校验改了回答文本时的回调（可选，阶段四）。
 *        收到即**整段替换**已显示的回答（token 事件只能追加，这是唯一能反映
 *        "剥除越界引用编号"的入口）。不传时行为与之前完全一致
 * @param signal  用于中止请求（可选）
 */
export async function streamChat(
  question: string,
  conversationId: number | null,
  onToken: (text: string) => void,
  onSources: (sources: Source[]) => void,
  onConversationId: (id: number) => void,
  onError: (msg: string) => void,
  onRewrite?: (q: string) => void,
  onClarification?: (c: ClarificationRequest) => void,
  onTrace?: (steps: TraceStep[]) => void,
  onMessageId?: (meta: PersistedMessageMeta) => void,
  onStreamEnd?: () => void,
  onCorrection?: (c: CorrectionPayload) => void,
  signal?: AbortSignal,
) {
  const res = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question, conversation_id: conversationId }),
    cache: 'no-store',
    signal,
  })

  if (!res.ok) {
    onError(`请求失败: ${res.status} ${res.statusText}`)
    return
  }
  if (!res.body) {
    onError('响应流为空')
    return
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    // 统一换行符（兼容 \r\n 和 \r），确保事件分割正常
    buffer = buffer.replace(/\r\n|\r/g, '\n')

    // SSE 协议: 以 \n\n 分隔事件
    let sep: number
    while ((sep = buffer.indexOf('\n\n')) !== -1) {
      const rawEvent = buffer.slice(0, sep)
      buffer = buffer.slice(sep + 2)

      // 解析 data: 行
      const lines = rawEvent.split('\n')
      const dataLines = lines
        .filter((l) => l.startsWith('data:'))
        .map((l) => l.slice(5).trimStart())
      if (dataLines.length === 0) continue

      const payload = dataLines.join('\n')
      try {
        const evt: SSEEvent = JSON.parse(payload)
        if (evt.type === 'token' && typeof evt.data === 'string') {
          onToken(evt.data)
        } else if (evt.type === 'rewrite' && typeof evt.data === 'string') {
          onRewrite?.(evt.data)
        } else if (evt.type === 'clarification_request' && evt.data && typeof evt.data === 'object') {
          onClarification?.(evt.data as ClarificationRequest)
        } else if (evt.type === 'trace' && Array.isArray(evt.data)) {
          onTrace?.(evt.data as TraceStep[])
        } else if (evt.type === 'correction' && evt.data && typeof evt.data === 'object') {
          onCorrection?.(evt.data as CorrectionPayload)
        } else if (evt.type === 'sources' && Array.isArray(evt.data)) {
          onSources(evt.data as Source[])
        } else if (evt.type === 'conversation_id' && typeof evt.data === 'number') {
          onConversationId(evt.data)
        } else if (evt.type === 'message_id' && evt.data && typeof evt.data === 'object') {
          // 兼容裸 id（老后端）：只给了 id 时 created_at 为 null，前端退回用本地时间
          const meta = evt.data as PersistedMessageMeta
          if (typeof meta.id === 'number') {
            onMessageId?.({ id: meta.id, created_at: meta.created_at ?? null })
          }
        } else if (evt.type === 'message_id' && typeof evt.data === 'number') {
          onMessageId?.({ id: evt.data, created_at: null })
        } else if (evt.type === 'error' && typeof evt.data === 'string') {
          onError(evt.data)
        }
      } catch (e) {
        console.error('SSE 解析失败:', e, payload)
      }
    }
  }

  // 流正常结束，通知调用方
  onStreamEnd?.()
}

/**
 * 触发数据入库 (POST /api/ingest)
 */
export async function ingestData(limitPerFile: number = 500) {
  const res = await fetch('/api/ingest', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ limit_per_file: limitPerFile }),
  })
  if (!res.ok) {
    throw new Error(`入库失败: ${res.status}`)
  }
  return res.json()
}

/**
 * 健康检查
 *
 * 必须 no-store：这个接口的唯一用途就是"点一下看后端还活着没"，
 * 任何形式的缓存命中都会让按钮变成假动作。
 */
export async function checkHealth() {
  const res = await fetch('/api/health', { cache: 'no-store' })
  if (!res.ok) throw new Error(`健康检查失败: ${res.status}`)
  return res.json()
}

// ===== 会话管理 API =====

/** 获取会话列表 */
export async function getConversations(): Promise<Conversation[]> {
  const res = await fetch('/api/conversations')
  if (!res.ok) throw new Error(`获取会话列表失败: ${res.status}`)
  const data = await res.json()
  return data.conversations
}

/** 创建新会话 */
export async function createConversation(title?: string): Promise<Conversation> {
  const res = await fetch('/api/conversations', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title }),
  })
  if (!res.ok) throw new Error(`创建会话失败: ${res.status}`)
  return res.json()
}

/** 删除会话 */
export async function deleteConversation(id: number): Promise<void> {
  const res = await fetch(`/api/conversations/${id}`, {
    method: 'DELETE',
  })
  if (!res.ok) throw new Error(`删除会话失败: ${res.status}`)
}

/** 更新会话标题 */
export async function updateConversation(id: number, title: string): Promise<Conversation> {
  const res = await fetch(`/api/conversations/${id}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title }),
  })
  if (!res.ok) throw new Error(`更新会话失败: ${res.status}`)
  return res.json()
}

/** 获取会话消息列表 */
export async function getConversationMessages(id: number): Promise<ChatMessageData[]> {
  const res = await fetch(`/api/conversations/${id}/messages`)
  if (!res.ok) throw new Error(`获取消息失败: ${res.status}`)
  const data = await res.json()
  return data.messages
}

/** 提交回答反馈（顶/踩 + 可选纠错文本）。后端幂等：同一消息重复提交视为更新。 */
export async function submitFeedback(
  messageId: number,
  thumbs: 'up' | 'down',
  correctedAnswer?: string,
  comment?: string,
): Promise<{ feedback_id: number; thumbs: string }> {
  const res = await fetch('/api/feedback', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      message_id: messageId,
      thumbs,
      corrected_answer: correctedAnswer || null,
      comment: comment || null,
    }),
    cache: 'no-store',
  })
  if (!res.ok) throw new Error(`反馈提交失败: ${res.status}`)
  return res.json()
}

/** 撤回反馈（撤销对某条助手消息的评价，供点错时反悔） */
export async function withdrawFeedback(messageId: number): Promise<void> {
  const res = await fetch(`/api/feedback/${messageId}`, {
    method: 'DELETE',
    cache: 'no-store',
  })
  if (!res.ok) throw new Error(`撤回反馈失败: ${res.status}`)
}

/** 单条差评（含反查出的原始问题与用户纠错） */
export interface FeedbackItem {
  feedback_id: number
  message_id: number
  question: string | null
  answer_head: string | null
  corrected_answer: string | null
  comment: string | null
  created_at: string | null
}

/** 反馈汇总 —— 人工介入的入口，见后端 GET /api/feedback/stats */
export interface FeedbackStats {
  total: number
  up: number
  down: number
  down_rate: number
  with_correction: number
  down_items: FeedbackItem[]
  /** 分页：当前页起点、是否还有下一页 */
  offset: number
  has_more: boolean
}

/**
 * 拉取差评汇总（分页）。
 *
 * 为什么需要它：`POST /api/feedback` 只把评价收进库，原来没有任何出口 ——
 * 用户点了 👎、甚至写了纠错，也从来没人读过。这个接口把差评连同
 * 「用户当时问的是什么」一起捞出来，反馈闭环才合上。
 */
export async function getFeedbackStats(
  limit = 50,
  offset = 0,
): Promise<FeedbackStats> {
  const res = await fetch(
    `/api/feedback/stats?limit=${limit}&offset=${offset}`,
    { cache: 'no-store' },
  )
  if (!res.ok) throw new Error(`获取反馈统计失败: ${res.status}`)
  return res.json()
}

// ===== 文档管理 API =====

export interface UploadedFile {
  id: number
  filename: string
  size_bytes: number
  size_mb: number
  extension: string
  file_path: string
  uploaded_at: string | null
}

export async function listUploads(): Promise<UploadedFile[]> {
  const res = await fetch('/api/uploads')
  if (!res.ok) throw new Error(`获取文档列表失败: ${res.status}`)
  const data = await res.json()
  return data.files
}

export async function uploadDocument(file: File): Promise<any> {
  const formData = new FormData()
  formData.append('file', file)
  const res = await fetch('/api/upload', {
    method: 'POST',
    body: formData,
  })
  if (!res.ok) throw new Error(`上传失败: ${res.status}`)
  return res.json()
}

/** 删除已上传文档（按 id 精确删除，后端会同时清理磁盘文件与已入库向量） */
export async function deleteUpload(id: number): Promise<void> {
  const res = await fetch(`/api/uploads/${id}`, {
    method: 'DELETE',
  })
  if (!res.ok) throw new Error(`删除失败: ${res.status}`)
}

// ===== SSE 流式入库 =====

export interface IngestProgress {
  type: 'start' | 'progress' | 'done' | 'error'
  total?: number
  done?: number
  progress?: number
  department?: string
  by_department?: Record<string, number>
  data?: string
}

export async function streamIngest(
  limitPerFile: number = 500,
  useCleaned: boolean = true,
  onProgress: (p: IngestProgress) => void,
  onError: (msg: string) => void,
  onDone: (total: number, byDept: Record<string, number>) => void,
) {
  const res = await fetch('/api/ingest-stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ limit_per_file: limitPerFile, use_cleaned: useCleaned }),
    cache: 'no-store',
  })

  if (!res.ok) {
    onError(`请求失败: ${res.status}`)
    return
  }
  if (!res.body) {
    onError('响应流为空')
    return
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder('utf-8')
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })
    buffer = buffer.replace(/\r\n|\r/g, '\n')

    let sep: number
    while ((sep = buffer.indexOf('\n\n')) !== -1) {
      const rawEvent = buffer.slice(0, sep)
      buffer = buffer.slice(sep + 2)

      const lines = rawEvent.split('\n')
      const dataLines = lines
        .filter((l) => l.startsWith('data:'))
        .map((l) => l.slice(5).trimStart())
      if (dataLines.length === 0) continue

      try {
        const evt: IngestProgress = JSON.parse(dataLines.join('\n'))
        if (evt.type === 'progress' || evt.type === 'start') {
          onProgress(evt)
        } else if (evt.type === 'done') {
          onDone(evt.total || 0, evt.by_department || {})
        } else if (evt.type === 'error' && evt.data) {
          onError(evt.data)
        }
      } catch (e) {
        console.error('SSE 解析失败:', e)
      }
    }
  }
}
