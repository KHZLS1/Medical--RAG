import { useEffect, useState } from 'react'
import Chat from './components/Chat'
import { checkHealth, ingestData } from './api/chat'

export default function App() {
  const [health, setHealth] = useState<string>('检测中...')
  const [ingesting, setIngesting] = useState(false)
  const [ingestMsg, setIngestMsg] = useState('')

  async function refreshHealth() {
    try {
      const r = await checkHealth()
      setHealth(`Milvus: ${r.milvus_uri} | collection: ${r.collection}`)
    } catch {
      setHealth('后端未响应，请先启动 backend')
    }
  }

  useEffect(() => {
    refreshHealth()
  }, [])

  async function handleIngest() {
    setIngesting(true)
    setIngestMsg('正在入库...')
    try {
      const r = await ingestData(500)
      setIngestMsg(
        `✅ 入库完成: 共 ${r.ingested} 条 | ` +
          Object.entries(r.by_department)
            .map(([k, v]) => `${k}=${v}`)
            .join(', '),
      )
    } catch (e) {
      setIngestMsg(`❌ 入库失败: ${(e as Error).message}`)
    } finally {
      setIngesting(false)
    }
  }

  return (
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
            onClick={handleIngest}
            disabled={ingesting}
            className="btn-primary"
          >
            {ingesting ? '入库中...' : '一键入库 500 条'}
          </button>
        </div>
      </header>

      <div className="disclaimer">
        ⚠️ 本系统回答仅基于公开医学问答数据集与大模型生成，仅供参考，<b>不能替代执业医师诊断</b>。
        涉及急症请立即拨打 120。
      </div>

      {ingestMsg && <div className="ingest-msg">{ingestMsg}</div>}

      <Chat />
    </div>
  )
}
