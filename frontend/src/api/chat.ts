// SSE 流式问答客户端
// 后端 /api/chat 用 POST + SSE 返回，前端用 fetch + ReadableStream 解析

export interface Source {
  index: number
  department: string
  title: string
  source: string
  snippet: string
}

interface SSEEvent {
  type: 'token' | 'sources' | 'error'
  data: string | Source[]
}

/**
 * 流式问答
 * @param question 用户问题
 * @param onToken  每收到一段文本时的回调
 * @param onSources 收到引用来源时的回调
 * @param onError  出错回调
 */
export async function streamChat(
  question: string,
  onToken: (text: string) => void,
  onSources: (sources: Source[]) => void,
  onError: (msg: string) => void,
) {
  const res = await fetch('/api/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
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
        } else if (evt.type === 'error' && typeof evt.data === 'string') {
          onError(evt.data)
        }
      } catch (e) {
        console.error('SSE 解析失败:', e, payload)
      }
    }
  }
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
