import { useEffect, useState, useMemo } from 'react'
import CopyButton from './CopyButton'
import Icon from './Icon'
import { getFeedbackStats, type FeedbackStats, type FeedbackItem } from '../api/chat'

/**
 * 一条差评拼成纯文本，供「复制」按钮写入剪贴板。
 *
 * 为什么要整条而不是只复制回答：这份数据的主要用途是**回流做标注**——
 * 拿到一条 badcase 得同时知道"他问的是什么""模型答了什么""人工纠正成什么"，
 * 拆开复制就丢了上下文。空字段直接不输出，免得贴出来一串"纠错："后面跟空。
 */
function formatFeedbackItem(it: FeedbackItem): string {
  const lines = [
    `问：${it.question || '（未能反查到原问题）'}`,
    `答：${it.answer_head || '（消息已删除）'}`,
  ]
  if (it.corrected_answer) lines.push(`纠错：${it.corrected_answer}`)
  if (it.comment) lines.push(`备注：${it.comment}`)
  lines.push(
    `#${it.feedback_id} · 消息 ${it.message_id} · ${it.created_at?.slice(0, 19) || ''}`,
  )
  return lines.join('\n')
}

/**
 * 反馈看板：把「用户点了差评」变成「有人会去看」。
 *
 * 动机很朴素：`POST /api/feedback` 上线后，评价能存进库，但**没有出口** ——
 * 要看得手动翻数据库，于是没人看。用户写的 `corrected_answer`（现成的标注数据）
 * 就这么白攒着。这个面板就是那条闭环的出口：
 *
 *     用户点差评 → 落库 → 看板列出「他当时问的是什么」→ 人工改 Prompt / 补语料
 *
 * 所以列表里**必须显示原始问题**，只给一条回答开头是没法定位问题的。
 */
export type TriageStatus = 'pending' | 'accepted' | 'ignored'
const FEEDBACK_STATUS_KEY = 'medical-rag.feedback-status'

function loadStatusMap(): Record<number, TriageStatus> {
  try {
    const raw = localStorage.getItem(FEEDBACK_STATUS_KEY)
    return raw ? JSON.parse(raw) : {}
  } catch {
    return {}
  }
}

function saveStatusMap(map: Record<number, TriageStatus>) {
  try {
    localStorage.setItem(FEEDBACK_STATUS_KEY, JSON.stringify(map))
  } catch {}
}

export default function FeedbackPanel({ onClose }: { onClose: () => void }) {
  const [stats, setStats] = useState<FeedbackStats | null>(null)
  const [items, setItems] = useState<FeedbackItem[]>([])
  const [offset, setOffset] = useState(0)
  const [hasMore, setHasMore] = useState(false)
  const [loading, setLoading] = useState(true)
  const [loadingMore, setLoadingMore] = useState(false)
  const [err, setErr] = useState('')
  const [statusMap, setStatusMap] = useState<Record<number, TriageStatus>>(loadStatusMap)
  const [filterMode, setFilterMode] = useState<'all' | 'pending' | 'accepted' | 'ignored' | 'correction'>('all')

  function setStatus(id: number, status: TriageStatus) {
    setStatusMap((prev) => {
      const next = { ...prev, [id]: status }
      saveStatusMap(next)
      return next
    })
  }

  function exportDataset() {
    const records = items.map((it) => ({
      feedback_id: it.feedback_id,
      message_id: it.message_id,
      prompt: it.question,
      chosen: it.corrected_answer || undefined,
      rejected: it.answer_head,
      audit_status: statusMap[it.feedback_id] || 'pending',
      comment: it.comment || undefined,
      created_at: it.created_at,
    }))

    const blob = new Blob([JSON.stringify(records, null, 2)], {
      type: 'application/json;charset=utf-8',
    })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `medical_feedback_dataset_${new Date().toISOString().slice(0, 10)}.json`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }

  const PAGE = 50

  async function load(append: boolean) {
    if (append) {
      setLoadingMore(true)
    } else {
      setLoading(true)
    }
    setErr('')
    try {
      const nextOffset = append ? offset : 0
      const s = await getFeedbackStats(PAGE, nextOffset)
      setStats(s)
      setItems((prev) => (append ? [...prev, ...s.down_items] : s.down_items))
      setOffset(s.offset + s.down_items.length)
      setHasMore(s.has_more)
    } catch (e) {
      setErr((e as Error).message)
    } finally {
      setLoading(false)
      setLoadingMore(false)
    }
  }

  useEffect(() => {
    load(false)
  }, [])

  // 抽屉的键盘出口：背景遮罩由 App 渲染，Esc 是它的等价操作
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  const filteredItems = useMemo(() => {
    return items.filter((it) => {
      const st = statusMap[it.feedback_id] || 'pending'
      if (filterMode === 'pending') return st === 'pending'
      if (filterMode === 'accepted') return st === 'accepted'
      if (filterMode === 'ignored') return st === 'ignored'
      if (filterMode === 'correction') return !!it.corrected_answer
      return true
    })
  }, [items, statusMap, filterMode])

  return (
    <aside className="sheet" role="dialog" aria-modal="true" aria-label="反馈看板">
      <div className="sheet-head">
        <Icon name="feedback" size="sm" />
        <h2>反馈看板</h2>
        <button className="btn btn-icon" type="button" aria-label="关闭" onClick={onClose}>
          <Icon name="x" size="sm" />
        </button>
      </div>

      <div className="sheet-body">
        <section className="sheet-section">
          <div className="sheet-section-header">
            <h3>汇总</h3>
            <button className="btn" type="button" onClick={() => load(false)} disabled={loading}>
              <Icon name="refresh" size="sm" />
              {loading ? '加载中…' : '刷新'}
            </button>
          </div>

          {err ? (
            <div className="doc-empty">
              <span className="icon-text">
                <Icon name="alert" size="sm" />
                无法加载反馈数据
              </span>
              <div className="fb-hint">后端未启动或接口不可用时会出现这个提示。</div>
            </div>
          ) : !stats ? (
            <div className="doc-empty">加载中…</div>
          ) : (
            <>
              <div className="fb-cards">
                <div className="fb-card">
                  <div className="fb-card-num">{stats.total}</div>
                  <div className="fb-card-label">总评价</div>
                </div>
                <div className="fb-card">
                  <div className="fb-card-num fb-up">{stats.up}</div>
                  <div className="fb-card-label">有用</div>
                </div>
                <div className="fb-card">
                  <div className="fb-card-num fb-down">{stats.down}</div>
                  <div className="fb-card-label">不准确</div>
                </div>
                <div className="fb-card">
                  <div className="fb-card-num">{(stats.down_rate * 100).toFixed(1)}%</div>
                  <div className="fb-card-label">差评率</div>
                </div>
                <div className="fb-card">
                  <div className="fb-card-num">{stats.with_correction}</div>
                  <div className="fb-card-label">带人工纠错</div>
                </div>
              </div>
              <div className="fb-hint">
                「带人工纠错」是用户自己写的正确答案，是最现成的标注数据 —— 优先看这批。
              </div>
            </>
          )}
        </section>

        {stats && (
          <section className="sheet-section">
            <div className="sheet-section-header">
              <h3>差评清单（{filteredItems.length} / {items.length}）</h3>
            </div>

            <div className="fb-toolbar">
              <div className="fb-filter-tabs">
                <button
                  type="button"
                  className={`fb-filter-tab${filterMode === 'all' ? ' is-active' : ''}`}
                  onClick={() => setFilterMode('all')}
                >
                  全部 ({items.length})
                </button>
                <button
                  type="button"
                  className={`fb-filter-tab${filterMode === 'pending' ? ' is-active' : ''}`}
                  onClick={() => setFilterMode('pending')}
                >
                  待审核 (
                  {
                    items.filter(
                      (it) => (statusMap[it.feedback_id] || 'pending') === 'pending',
                    ).length
                  }
                  )
                </button>
                <button
                  type="button"
                  className={`fb-filter-tab${filterMode === 'accepted' ? ' is-active' : ''}`}
                  onClick={() => setFilterMode('accepted')}
                >
                  已采纳 (
                  {
                    items.filter(
                      (it) => statusMap[it.feedback_id] === 'accepted',
                    ).length
                  }
                  )
                </button>
                <button
                  type="button"
                  className={`fb-filter-tab${filterMode === 'correction' ? ' is-active' : ''}`}
                  onClick={() => setFilterMode('correction')}
                >
                  仅含纠错 ({items.filter((it) => !!it.corrected_answer).length})
                </button>
              </div>

              <button
                type="button"
                className="btn btn-sm"
                title="导出为标准微调与评测数据集 (JSON)"
                disabled={items.length === 0}
                onClick={exportDataset}
              >
                <Icon name="upload" size="sm" />
                导出标注集
              </button>
            </div>

            {filteredItems.length === 0 ? (
              <div className="doc-empty">
                {items.length === 0
                  ? stats.total === 0
                    ? '还没有收到任何反馈'
                    : '目前没有差评'
                  : '当前筛选分类下暂无反馈'}
              </div>
            ) : (
              <>
                <div className="doc-list">
                  {filteredItems.map((it) => {
                    const st = statusMap[it.feedback_id] || 'pending'
                    return (
                      <div key={it.feedback_id} className="fb-item">
                        <div className="fb-q">
                          <span className="fb-tag">问</span>
                          <span style={{ flex: 1 }}>{it.question || '（未能反查到原问题）'}</span>
                          {it.corrected_answer && (
                            <span className="fb-tag-correction">✨ 包含纠错</span>
                          )}
                        </div>
                        <div className="fb-a">
                          <span className="fb-tag fb-tag-a">答</span>
                          {it.answer_head || '（消息已删除）'}
                        </div>
                        {it.corrected_answer && (
                          <div className="fb-correct">
                            <span className="fb-tag fb-tag-c">纠错</span>
                            {it.corrected_answer}
                          </div>
                        )}
                        {it.comment && <div className="fb-meta">备注：{it.comment}</div>}
                        <div className="fb-item-foot">
                          <div className="fb-status-actions">
                            <button
                              type="button"
                              className={`fb-status-btn${st === 'pending' ? ' is-active' : ''}`}
                              onClick={() => setStatus(it.feedback_id, 'pending')}
                            >
                              待审
                            </button>
                            <button
                              type="button"
                              className={`fb-status-btn${st === 'accepted' ? ' is-accepted' : ''}`}
                              onClick={() => setStatus(it.feedback_id, 'accepted')}
                            >
                              ✓ 采纳
                            </button>
                            <button
                              type="button"
                              className={`fb-status-btn${st === 'ignored' ? ' is-ignored' : ''}`}
                              onClick={() => setStatus(it.feedback_id, 'ignored')}
                            >
                              ✕ 忽略
                            </button>
                          </div>
                          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                            <span className="fb-meta">
                              #{it.feedback_id} · 消息 {it.message_id} ·{' '}
                              {it.created_at?.slice(0, 19) || ''}
                            </span>
                            <CopyButton
                              text={formatFeedbackItem(it)}
                              label="复制"
                              className="copy-mini"
                            />
                          </div>
                        </div>
                      </div>
                    )
                  })}
                </div>
                {hasMore && (
                  <button
                    className="btn fb-load-more"
                    type="button"
                    onClick={() => load(true)}
                    disabled={loadingMore}
                  >
                    {loadingMore ? '加载中…' : '加载更多'}
                  </button>
                )}
              </>
            )}
          </section>
        )}
      </div>
    </aside>
  )
}