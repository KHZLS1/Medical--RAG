import { useEffect, useState } from 'react'
import { getFeedbackStats, type FeedbackStats } from '../api/chat'

/**
 * 反馈看板：把「用户点了 👎」变成「有人会去看」。
 *
 * 动机很朴素：`POST /api/feedback` 上线后，评价能存进库，但**没有出口** ——
 * 要看得手动翻数据库，于是没人看。用户写的 `corrected_answer`（现成的标注数据）
 * 就这么白攒着。这个面板就是那条闭环的出口：
 *
 *     用户点 👎 → 落库 → 看板列出「他当时问的是什么」→ 人工改 Prompt / 补语料
 *
 * 所以列表里**必须显示原始问题**，只给一条回答开头是没法定位问题的。
 */
export default function FeedbackPanel({ onClose }: { onClose: () => void }) {
  const [stats, setStats] = useState<FeedbackStats | null>(null)
  const [loading, setLoading] = useState(true)
  const [err, setErr] = useState('')

  async function load() {
    setLoading(true)
    setErr('')
    try {
      setStats(await getFeedbackStats(50))
    } catch (e) {
      setErr((e as Error).message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    load()
  }, [])

  return (
    <div className="doc-panel-overlay" onClick={onClose}>
      <div className="doc-panel" onClick={(e) => e.stopPropagation()}>
        <div className="doc-panel-header">
          <h2>💬 反馈看板</h2>
          <button className="doc-panel-close" onClick={onClose}>✕</button>
        </div>

        <div className="doc-panel-body">
          <div className="doc-section">
            <div className="doc-section-header">
              <h3>汇总</h3>
              <button className="btn-small" onClick={load} disabled={loading}>
                {loading ? '加载中...' : '刷新'}
              </button>
            </div>

            {err ? (
              <div className="doc-empty">
                ❌ {err}
                <div className="fb-hint">后端未启动或接口不可用时会出现这个提示。</div>
              </div>
            ) : !stats ? (
              <div className="doc-empty">加载中...</div>
            ) : (
              <>
                <div className="fb-cards">
                  <div className="fb-card">
                    <div className="fb-card-num">{stats.total}</div>
                    <div className="fb-card-label">总评价</div>
                  </div>
                  <div className="fb-card">
                    <div className="fb-card-num fb-up">{stats.up}</div>
                    <div className="fb-card-label">👍 有用</div>
                  </div>
                  <div className="fb-card">
                    <div className="fb-card-num fb-down">{stats.down}</div>
                    <div className="fb-card-label">👎 不准确</div>
                  </div>
                  <div className="fb-card">
                    <div className="fb-card-num">
                      {(stats.down_rate * 100).toFixed(1)}%
                    </div>
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
          </div>

          {stats && (
            <div className="doc-section">
              <h3>差评清单 ({stats.down_items.length})</h3>
              {stats.down_items.length === 0 ? (
                <div className="doc-empty">
                  {stats.total === 0
                    ? '还没有收到任何反馈'
                    : '目前没有差评'}
                </div>
              ) : (
                <div className="doc-list">
                  {stats.down_items.map((it) => (
                    <div key={it.feedback_id} className="fb-item">
                      <div className="fb-q">
                        <span className="fb-tag">问</span>
                        {it.question || '（未能反查到原问题）'}
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
                      {it.comment && (
                        <div className="fb-meta">备注：{it.comment}</div>
                      )}
                      <div className="fb-meta">
                        #{it.feedback_id} · 消息 {it.message_id} ·{' '}
                        {it.created_at?.slice(0, 19) || ''}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
