import { useEffect, useRef, useState } from 'react'
import CopyButton from './CopyButton'
import Icon from './Icon'
import type { Source, TraceStep } from '../api/chat'

/** 证据栏要显示的那一轮的全部材料，由 Chat 从当前选中的助手消息里提取 */
export interface EvidencePayload {
  /** 该助手消息在消息数组里的下标，用作证据卡的 key 前缀 */
  msgIndex: number
  /** 「关联第 N 轮回答」这行说明文字 */
  scope: string
  sources: Source[]
  rewrite?: string
  trace?: TraceStep[]
  /** 这一轮还在流式生成：来源块先不渲染，免得闪一个"0 条"出来 */
  streaming: boolean
  /** 证据不足（澄清/兜底）：来源块换成解释文案，而不是一片空白 */
  noEvidence: boolean
}

export type EvidenceBlockKey = 'rewrite' | 'trace' | 'sources'

/**
 * 决策类节点：混合检索/精排/忠实性校验/证据不足兜底。
 * 这些步决定了"用不用资料、给不给答案"，在时间轴上点成实心点，
 * 让排查 badcase 时一眼能找到分叉处。
 */
const KEY_STEPS = new Set(['混合检索', '精排', '忠实性校验', '证据不足兜底', '人工澄清'])

/** 把一步 trace 的明细键值铺成一行（step 之外的非空字段）。 */
function traceDetail(t: TraceStep): string {
  return Object.entries(t)
    .filter(([k, v]) => k !== 'step' && v !== '' && v != null)
    .map(([k, v]) => `${k}: ${String(v)}`)
    .join('  ·  ')
}

/** 把「检索过程」整条链路摊成纯文本，便于贴进反馈/工单。 */
function formatTrace(steps: TraceStep[]): string {
  return steps
    .map((t, i) => {
      const detail = traceDetail(t)
      return `${i + 1}. ${t.step}${detail ? ` — ${detail}` : ''}`
    })
    .join('\n')
}

/**
 * 来源卡片「复制出处」的文本。
 *
 * 只带出处的三要素（编号 / 标题 / 科室）+ 文件路径，**不带正文片段**：
 * 片段和完整原文各有自己的复制入口，混在一起会让"复制出处"变成复制一大段。
 * 做文献标注时需要的正是这三要素，路径则用于溯源。
 */
function formatSourceCitation(s: Source): string {
  return `[${s.index}] ${s.title}（${s.department || '未知科室'}）\n来源：${s.source}`
}

export default function EvidencePanel({
  payload,
  activeCite,
  openBlocks,
  onToggleBlock,
  onSelectCite,
  onClose,
}: {
  payload: EvidencePayload | null
  activeCite: number | null
  openBlocks: Record<EvidenceBlockKey, boolean>
  onToggleBlock: (k: EvidenceBlockKey) => void
  onSelectCite: (n: number) => void
  onClose: () => void
}) {
  const [openFull, setOpenFull] = useState<Record<number, boolean>>({})
  const cardRefs = useRef<Record<number, HTMLElement | null>>({})

  // 点正文角标后把对应证据卡滚到视野中间。放在这里而不是点击回调里，
  // 是因为点「证据栏查看」只切轮次、不定位，滚动必须由 activeCite 变化驱动。
  useEffect(() => {
    if (activeCite == null) return
    const el = cardRefs.current[activeCite]
    if (el) el.scrollIntoView({ behavior: 'smooth', block: 'center' })
  }, [activeCite, payload?.msgIndex])

  const sources = payload?.sources ?? []

  // 换一轮就收起上一轮展开过的原文：证据编号每轮都从 1 重排，
  // 不重置的话新一轮的第 1 条会"继承"上一轮的展开态。
  useEffect(() => {
    setOpenFull({})
  }, [payload?.msgIndex])

  return (
    <aside className="evidence" aria-label="证据链">
      <div className="evidence-head">
        <div className="evidence-head-row">
          <h2>
            <Icon name="link" size="sm" />
            证据链
          </h2>
          <span className="count-pill">{sources.length} 条</span>
          <button className="mini" type="button" aria-label="收起证据栏" onClick={onClose}>
            <Icon name="x" size="sm" />
          </button>
        </div>
        <div className="evidence-scope">
          {payload ? payload.scope : '点击正文角标可切换查看'}
        </div>
      </div>

      <div className="evidence-scroll">
        {!payload ? (
          <div className="ev-empty">
            <strong>尚未选择回答</strong>
            点击任一回答下方的「证据栏查看」，或直接点正文里的编号角标，
            这里会显示那一轮的检索词、检索链路与来源原文。
          </div>
        ) : (
          <>
            {payload.rewrite && (
              <section className="block">
                <button
                  className="block-head"
                  type="button"
                  aria-expanded={openBlocks.rewrite}
                  onClick={() => onToggleBlock('rewrite')}
                >
                  <Icon name="search" size="sm" />
                  实际检索词
                  <Icon name="chev" size="sm" className="chev" />
                </button>
                <div className="block-body">
                  <div className="rewrite-text">{payload.rewrite}</div>
                  <div className="block-actions">
                    <CopyButton text={payload.rewrite} label="复制检索词" className="copy-mini" />
                  </div>
                </div>
              </section>
            )}

            {payload.trace && payload.trace.length > 0 && (
              <section className="block">
                <button
                  className="block-head"
                  type="button"
                  aria-expanded={openBlocks.trace}
                  onClick={() => onToggleBlock('trace')}
                >
                  <Icon name="route" size="sm" />
                  检索过程
                  <Icon name="chev" size="sm" className="chev" />
                </button>
                <div className="block-body">
                  <ol className="trace">
                    {payload.trace.map((t, i) => (
                      <li key={i} className={KEY_STEPS.has(t.step) ? 'is-key' : undefined}>
                        <span className="trace-step">{t.step}</span>
                        <span className="trace-detail">{traceDetail(t)}</span>
                      </li>
                    ))}
                  </ol>
                  <div className="block-actions">
                    <CopyButton
                      text={formatTrace(payload.trace)}
                      label="复制检索过程"
                      className="copy-mini"
                    />
                  </div>
                </div>
              </section>
            )}

            {payload.streaming && sources.length === 0 ? null : payload.noEvidence ? (
              <div className="ev-empty">
                <strong>本轮未检索到可用证据</strong>
                系统判定证据不足，已中断生成并向你追问。补充信息后会重新检索，
                来源会出现在这里。
              </div>
            ) : (
              <section className="block">
                <button
                  className="block-head"
                  type="button"
                  aria-expanded={openBlocks.sources}
                  onClick={() => onToggleBlock('sources')}
                >
                  <Icon name="link" size="sm" />
                  引用来源 · {sources.length} 条
                  <Icon name="chev" size="sm" className="chev" />
                </button>
                <div className="block-body">
                  <div className="ev-list">
                    {sources.map((s) => {
                      const isOpen = !!openFull[s.index]
                      return (
                        <article
                          key={s.index}
                          ref={(el) => {
                            cardRefs.current[s.index] = el
                          }}
                          className={`ev-card${activeCite === s.index ? ' is-active' : ''}`}
                          onClick={(e) => {
                            // 卡内的按钮（查看原文 / 复制出处）有自己的动作，别顺手把卡也选中
                            if ((e.target as HTMLElement).closest('button')) return
                            onSelectCite(s.index)
                          }}
                        >
                          <div className="ev-top">
                            <span className="ev-index">{s.index}</span>
                            <span className="ev-title">{s.title}</span>
                            <span className="dept-tag">{s.department || '未知科室'}</span>
                          </div>
                          <p className="ev-snippet">{s.snippet}</p>
                          {s.full_text && (
                            <>
                              <button
                                className={`mini${isOpen ? ' is-ok' : ''}`}
                                type="button"
                                onClick={(e) => {
                                  e.stopPropagation()
                                  setOpenFull((prev) => ({ ...prev, [s.index]: !prev[s.index] }))
                                }}
                              >
                                <Icon name="file" size="sm" />
                                {isOpen ? '收起原文' : '查看完整原文'}
                              </button>
                              {isOpen && <div className="ev-full">{s.full_text}</div>}
                            </>
                          )}
                          <div className="ev-foot">
                            <span className="ev-path" title={s.source}>
                              {s.source}
                            </span>
                            <CopyButton
                              text={formatSourceCitation(s)}
                              label="复制出处"
                              className="copy-mini"
                            />
                          </div>
                        </article>
                      )
                    })}
                  </div>
                </div>
              </section>
            )}
          </>
        )}
      </div>
    </aside>
  )
}