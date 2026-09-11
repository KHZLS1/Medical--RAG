import { useState, useEffect, useRef } from 'react'
import {
  listUploads,
  uploadDocument,
  deleteUpload,
  streamIngest,
  type UploadedFile,
  type IngestProgress,
} from '../api/chat'

export default function DocumentPanel({ onClose }: { onClose: () => void }) {
  const [files, setFiles] = useState<UploadedFile[]>([])
  const [uploading, setUploading] = useState(false)
  const [ingesting, setIngesting] = useState(false)
  const [progress, setProgress] = useState<IngestProgress | null>(null)
  const [ingestMsg, setIngestMsg] = useState('')
  const [ingestLimit, setIngestLimit] = useState(500)
  const [useCleaned, setUseCleaned] = useState(true)
  const fileInputRef = useRef<HTMLInputElement>(null)

  async function loadFiles() {
    try {
      const list = await listUploads()
      setFiles(list)
    } catch (e) {
      console.error('加载文档列表失败:', e)
    }
  }

  useEffect(() => {
    loadFiles()
  }, [])

  async function handleUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (!file) return
    setUploading(true)
    try {
      await uploadDocument(file)
      await loadFiles()
    } catch (e) {
      console.error('上传失败:', e)
    } finally {
      setUploading(false)
      if (fileInputRef.current) fileInputRef.current.value = ''
    }
  }

  async function handleDelete(target: UploadedFile) {
    if (!confirm(`确定删除 ${target.filename} 吗？\n（同时会移除该文档已入库的向量）`)) return
    try {
      await deleteUpload(target.id)
      setFiles((prev) => prev.filter((f) => f.id !== target.id))
    } catch (e) {
      console.error('删除失败:', e)
    }
  }

  async function handleIngest() {
    setIngesting(true)
    setProgress(null)
    setIngestMsg('正在准备入库...')
    try {
      await streamIngest(
        ingestLimit,
        useCleaned,
        (p) => setProgress(p),
        (msg) => setIngestMsg(`❌ ${msg}`),
        (total, byDept) => {
          const deptStr = Object.entries(byDept)
            .map(([k, v]) => `${k}=${v}`)
            .join(', ')
          setIngestMsg(`✅ 入库完成: 共 ${total} 条 | ${deptStr}`)
        },
      )
    } catch (e) {
      setIngestMsg(`❌ 入库失败: ${(e as Error).message}`)
    } finally {
      setIngesting(false)
    }
  }

  return (
    <div className="doc-panel-overlay" onClick={onClose}>
      <div className="doc-panel" onClick={(e) => e.stopPropagation()}>
        <div className="doc-panel-header">
          <h2>📄 文档管理</h2>
          <button className="doc-panel-close" onClick={onClose}>✕</button>
        </div>

        <div className="doc-panel-body">
          {/* 上传区域 */}
          <div className="doc-section">
            <h3>上传文档</h3>
            <div className="upload-area" onClick={() => fileInputRef.current?.click()}>
              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf,.docx,.txt,.md,.csv"
                onChange={handleUpload}
                style={{ display: 'none' }}
              />
              <div className="upload-placeholder">
                {uploading ? '上传中...' : '📎 点击选择文件 (PDF/Word/TXT/MD/CSV)'}
              </div>
            </div>
          </div>

          {/* 已上传文档列表 */}
          <div className="doc-section">
            <div className="doc-section-header">
              <h3>已上传文档 ({files.length})</h3>
              <button className="btn-small" onClick={loadFiles}>刷新</button>
            </div>
            {files.length === 0 ? (
              <div className="doc-empty">暂无上传文档</div>
            ) : (
              <div className="doc-list">
                {files.map((f) => (
                  <div key={f.id} className="doc-item">
                    <div className="doc-item-info">
                      <span className="doc-item-name">{f.filename}</span>
                      <span className="doc-item-meta">
                        {f.extension} · {f.size_mb}MB · {f.uploaded_at?.slice(0, 19) || ''}
                      </span>
                    </div>
                    <button
                      className="doc-item-delete"
                      onClick={() => handleDelete(f)}
                    >
                      删除
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* 批量入库区域 */}
          <div className="doc-section">
            <h3>批量数据入库</h3>
            <div className="ingest-controls">
              <label className="ingest-label">
                每科室条数:
                <input
                  type="number"
                  value={ingestLimit}
                  onChange={(e) => setIngestLimit(Number(e.target.value))}
                  min={1}
                  disabled={ingesting}
                  className="ingest-input"
                />
              </label>
              <label className="ingest-label">
                <input
                  type="checkbox"
                  checked={useCleaned}
                  onChange={(e) => setUseCleaned(e.target.checked)}
                  disabled={ingesting}
                />
                使用清洗后数据
              </label>
              <button
                className="btn-primary"
                onClick={handleIngest}
                disabled={ingesting}
              >
                {ingesting ? '入库中...' : '开始入库'}
              </button>
            </div>

            {progress && (
              <div className="ingest-progress">
                <div className="progress-bar-container">
                  <div
                    className="progress-bar"
                    style={{ width: `${progress.progress || 0}%` }}
                  />
                </div>
                <div className="progress-text">
                  {progress.done}/{progress.total} 条 ({progress.progress}%)
                  {progress.department && ` · 当前科室: ${progress.department}`}
                </div>
              </div>
            )}

            {ingestMsg && <div className="ingest-msg">{ingestMsg}</div>}
          </div>
        </div>
      </div>
    </div>
  )
}
