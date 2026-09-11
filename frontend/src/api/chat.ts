// SSE 流式问答客户端 + 会话管理 API
// 后端 /api/chat 用 POST + SSE 返回，前端用 fetch + ReadableStream 解析

export interface Source {
  index: number
  department: string
  title: string
  source: string
  snippet: string
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
}

interface SSEEvent {
  type: 'token' | 'sources' | 'error' | 'conversation_id'
  data: string | Source[] | number
}

/**
 * 流式问答（支持多轮上下文）
 * @param question 用户问题
 * @param conversationId 会话ID（可选，不传则后端自动新建会话）
 * @param onToken  每收到一段文本时的回调
 * @param onSources 收到引用来源时的回调
 * @param onConversationId 收到会话ID时的回调（新建会话时触发）
 * @param onError  出错回调
 */
export async function streamChat(
  question: string,
  conversationId: number | null,
  onToken: (text: string) => void,
  onSources: (sources: Source[]) => void,
  onConversationId: (id: number) => void,
  onError: (msg: string) => void,
  onStreamEnd?: () => void,
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
        } else if (evt.type === 'sources' && Array.isArray(evt.data)) {
          onSources(evt.data as Source[])
        } else if (evt.type === 'conversation_id' && typeof evt.data === 'number') {
          onConversationId(evt.data)
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
 */
export async function checkHealth() {
  const res = await fetch('/api/health')
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
