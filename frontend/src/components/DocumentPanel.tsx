import { useState, useEffect, useRef } from 'react'
import {
  listUploads,
  uploadDocument,
  deleteUpload,
  streamIngest,
  type UploadedFile,
  type IngestProgress,
} from '../api/chat'
import CopyButton from './CopyButton'
import Icon from './Icon'

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

  // 抽屉的键盘出口：背景遮罩由 App 渲染，Esc 是它的等价操作
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

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
        (msg) => setIngestMsg(`入库失败: ${msg}`),
        (total, byDept) => {
          const deptStr = Object.entries(byDept)
            .map(([k, v]) => `${k}=${v}`)
            .join(', ')
          setIngestMsg(`入库完成: 共 ${total} 条 | ${deptStr}`)
        },
      )
    } catch (e) {
      setIngestMsg(`入库失败: ${(e as Error).message}`)
    } finally {
      setIngesting(false)
    }
  }

  return (
    <aside className="sheet" role="dialog" aria-modal="true" aria-label="文档管理">
      <div className="sheet-head">
        <Icon name="file" size="sm" />
        <h2>文档管理</h2>
        <button className="btn btn-icon" type="button" aria-label="关闭" onClick={onClose}>
          <Icon name="x" size="sm" />
        </button>
      </div>

      <div className="sheet-body">
        {/* 上传区域 */}
        <section className="sheet-section">
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
              <Icon name="upload" size="sm" />
              {uploading ? '上传中…' : '点击选择文件（PDF / Word / TXT / MD / CSV）'}
            </div>
          </div>
        </section>

        {/* 已上传文档列表 */}
        <section className="sheet-section">
          <div className="sheet-section-header">
            <h3>已上传文档（{files.length}）</h3>
            <button className="btn" type="button" onClick={loadFiles}>
              <Icon name="refresh" size="sm" />
              刷新
            </button>
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
                  <div className="doc-item-actions">
                    <CopyButton text={f.filename} label="复制文件名" className="copy-mini" />
                    <button className="doc-item-delete" type="button" onClick={() => handleDelete(f)}>
                      删除
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </section>

        {/* 批量入库区域 */}
        <section className="sheet-section">
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
            <button className="btn btn-accent" type="button" onClick={handleIngest} disabled={ingesting}>
              {ingesting ? '入库中…' : '开始入库'}
            </button>
          </div>

          {progress && (
            <div className="ingest-progress">
              <div className="progress-bar-container">
                <div className="progress-bar" style={{ width: `${progress.progress || 0}%` }} />
              </div>
              <div className="progress-text">
                {progress.done}/{progress.total} 条（{progress.progress}%）
                {progress.department && ` · 当前科室: ${progress.department}`}
              </div>
            </div>
          )}

          {ingestMsg && <div className="ingest-msg">{ingestMsg}</div>}
        </section>
      </div>
    </aside>
  )
}