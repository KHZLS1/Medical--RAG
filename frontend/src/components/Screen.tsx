import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import Icon from './Icon'
import { fetchOverview, type OverviewResult } from '../api/stats'

/**
 * 展厅数据大屏。
 *
 * 用途决定形态：这块屏是**无人值守轮播**的，所以
 *   1. 五屏自动轮转，每屏 14s，底部进度条可见，点圆点可跳；
 *   2. 真·浏览器全屏（requestFullscreen）—— 浏览器要求必须由用户手势触发，
 *      所以进屏后第一次点击/按键时自动尝试，同时右上角留手动开关；
 *      被拒绝也不影响：布局本身就是 position:fixed 铺满视口，只是浏览器外壳还在。
 *   3. 接口拉不到就退到离线快照（见 api/stats.ts），角上明确标注数据来源与时间。
 *
 * 数字口径：语料库规模（磁盘）与已入库向量（Milvus）是两回事，必须分开标注 ——
 * 小批量入库时两者差很远，把前者当后者展示就是在展厅里报假数字。
 */

const SLIDE_MS = 14000
const REFRESH_MS = 120000

const nf = new Intl.NumberFormat('zh-CN')

const fmtInt = (v: number | null | undefined) => (v == null ? '—' : nf.format(v))
const fmtPct = (v: number | null | undefined, digits = 1) =>
  v == null ? '—' : `${(v * 100).toFixed(digits)}%`
const fmtFixed = (v: number | null | undefined, digits = 2) =>
  v == null ? '—' : v.toFixed(digits)

function fmtTime(iso: string | null | undefined): string {
  if (!iso) return '—'
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return iso
  const p = (n: number) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`
}

interface Slide {
  key: string
  nav: string
  eyebrow: string
  title: string
  lead: string
  body: ReactNode
}

function Metric({
  value,
  unit,
  label,
  hint,
  tone,
}: {
  value: string
  unit?: string
  label: string
  hint?: string
  tone?: 'proof' | 'warn' | 'accent'
}) {
  return (
    <div className={`sc-metric${tone ? ` is-${tone}` : ''}`}>
      <div className="sc-metric-v">
        <span className="sc-metric-n">{value}</span>
        {unit && <span className="sc-metric-u">{unit}</span>}
      </div>
      <div className="sc-metric-l">{label}</div>
      {hint && <div className="sc-metric-h">{hint}</div>}
    </div>
  )
}

export default function Screen({ onExit }: { onExit: () => void }) {
  const [state, setState] = useState<OverviewResult | null>(null)
  const [index, setIndex] = useState(0)
  const [paused, setPaused] = useState(false)
  const [isFull, setIsFull] = useState(() => document.fullscreenElement !== null)
  const [fsHint, setFsHint] = useState(true)

  const load = useCallback(async () => {
    setState(await fetchOverview(10))
  }, [])

  useEffect(() => {
    load()
    const t = setInterval(load, REFRESH_MS)
    return () => clearInterval(t)
  }, [load])

  // ---- 全屏：只能由用户手势触发，所以进屏后第一次交互时尝试一次 ----
  useEffect(() => {
    function onFsChange() {
      setIsFull(document.fullscreenElement !== null)
    }
    document.addEventListener('fullscreenchange', onFsChange)
    return () => document.removeEventListener('fullscreenchange', onFsChange)
  }, [])

  useEffect(() => {
    if (isFull) return
    function tryFs() {
      setFsHint(false)
      document.documentElement.requestFullscreen().catch(() => {
        // 被拒绝（无手势 / 浏览器不支持 / 权限策略）：保持全视口布局即可，不必打扰用户
      })
    }
    window.addEventListener('pointerdown', tryFs, { once: true })
    window.addEventListener('keydown', tryFs, { once: true })
    return () => {
      window.removeEventListener('pointerdown', tryFs)
      window.removeEventListener('keydown', tryFs)
    }
  }, [isFull])

  // 离开大屏时收掉全屏，否则回到工作台会卡在全屏里
  useEffect(() => {
    return () => {
      if (document.fullscreenElement) document.exitFullscreen().catch(() => {})
    }
  }, [])

  const data = state?.data
  const corpus = data?.corpus
  const evaluation = data?.evaluation ?? null
  const usage = data?.usage ?? null
  const feedback = data?.feedback ?? null

  const slides = useMemo<Slide[]>(() => {
    if (!data || !corpus) return []

    const maxCount = Math.max(1, ...corpus.top_departments.map((d) => d.count))
    const gap =
      corpus.ingested == null ? null : corpus.total_records - corpus.ingested
    const coverageShare =
      corpus.total_records > 0 ? corpus.other_records / corpus.total_records : 0
    const restDepts = corpus.departments - corpus.top_departments.length

    const corpusSlide: Slide = {
      key: 'corpus',
      nav: '语料规模',
      eyebrow: '01 / 语料规模',
      title: '语料库与向量库',
      lead: '清洗后的公开医学对话语料，按科室拆分入库；右侧数字为 Milvus 中真正可被检索到的条数。',
      body: (
        <>
          <div className="sc-metrics">
            <Metric
              value={fmtInt(corpus.total_records)}
              unit="条"
              label="清洗后语料"
              hint="磁盘语料库规模"
              tone="accent"
            />
            <Metric
              value={fmtInt(corpus.departments)}
              unit="个"
              label="覆盖科室与病种"
              hint="按科室拆分文件"
            />
            <Metric
              value={corpus.ingested == null ? '未连接' : fmtInt(corpus.ingested)}
              unit={corpus.ingested == null ? undefined : '条'}
              label="已入库向量"
              hint="Milvus 实际可检索条数"
              tone={corpus.ingested == null ? 'warn' : 'proof'}
            />
          </div>
          <div className="sc-note">
            {gap == null ? (
              <>
                <b>向量库未连接</b>，当前只展示磁盘语料规模；入库条数需 Milvus 就绪后才可读。
              </>
            ) : gap === 0 ? (
              <>
                语料库与向量库<b>条数一致</b>，全量语料均已入库，无遗漏。
              </>
            ) : gap > 0 && gap < corpus.total_records * 0.01 ? (
              <>
                语料库（磁盘清洗产物）与向量库（可检索条数）是两个口径，当前相差{' '}
                <b>{fmtInt(gap)} 条</b>，属正常入库误差。
              </>
            ) : (
              <>
                <b>向量库与语料库相差 {fmtInt(gap)} 条</b>，可能为小批量入库（
                <code>--limit</code>）所致，检索覆盖面小于语料规模。
              </>
            )}
            {corpus.failed_files > 0 && (
              <span className="sc-warn-inline">
                <Icon name="alert" size="sm" />
                {corpus.failed_files} 个语料文件读取失败，未计入统计
              </span>
            )}
          </div>
        </>
      ),
    }

    const deptSlide: Slide = {
      key: 'departments',
      nav: '科室分布',
      eyebrow: '02 / 覆盖分布',
      title: '科室语料分布 · Top 10',
      lead: `按条数排序的十个科室；其余 ${restDepts} 个科室合计 ${fmtInt(corpus.other_records)} 条，占 ${fmtPct(coverageShare)}。`,
      body: (
        <div className="sc-bars">
          {corpus.top_departments.map((d, i) => (
            <div className="sc-bar" key={d.department}>
              <span className="sc-bar-rank">{String(i + 1).padStart(2, '0')}</span>
              <span className="sc-bar-name" title={d.department}>
                {d.department}
              </span>
              <span className="sc-bar-track">
                <span
                  className="sc-bar-fill"
                  style={{ width: `${(d.count / maxCount) * 100}%` }}
                />
              </span>
              <span className="sc-bar-num">{fmtInt(d.count)}</span>
              <span className="sc-bar-share">{fmtPct(d.count / corpus.total_records)}</span>
            </div>
          ))}
        </div>
      ),
    }

    const r = data.retrieval
    const recall = r.recall_k
    const retrievalSlide: Slide = {
      key: 'retrieval',
      nav: '检索链路',
      eyebrow: '03 / 检索链路',
      title: '混合召回 + 精排',
      lead: '稀疏与稠密两路并行召回，加权融合后交给 Cross-Encoder 精排，只保留少数几条作为作答依据。',
      body: (
        <>
          <div className="sc-flow">
            <div className="sc-flow-step">
              <span className="sc-flow-n">01</span>
              <Icon name="search" />
              <span className="sc-flow-t">改写补全</span>
              <span className="sc-flow-d">多轮里的指代与省略，结合上文补齐后再检索</span>
            </div>
            <div className="sc-flow-step">
              <span className="sc-flow-n">02</span>
              <Icon name="route" />
              <span className="sc-flow-t">BM25 稀疏召回</span>
              <span className="sc-flow-d">
                权重 {fmtFixed(r.bm25_weight)}
                {recall != null && <> · 召回 Top-{recall}</>}
              </span>
            </div>
            <div className="sc-flow-step">
              <span className="sc-flow-n">03</span>
              <Icon name="route" />
              <span className="sc-flow-t">向量稠密召回</span>
              <span className="sc-flow-d">
                权重 {fmtFixed(r.vector_weight)}
                {recall != null && <> · 召回 Top-{recall}</>}
              </span>
            </div>
            <div className="sc-flow-step">
              <span className="sc-flow-n">04</span>
              <Icon name="panel" />
              <span className="sc-flow-t">Cross-Encoder 精排</span>
              <span className="sc-flow-d">逐条重打相关性分，保留 Top-{r.reranker_top_k}</span>
            </div>
            <div className="sc-flow-step is-proof">
              <span className="sc-flow-n">05</span>
              <Icon name="link" />
              <span className="sc-flow-t">引用作答</span>
              <span className="sc-flow-d">严格限于资料范围，逐条标注 [1][2] 编号</span>
            </div>
          </div>
          <div className="sc-note">
            当前权重来自{' '}
            <b>{r.weights_source === 'tuned' ? '离线调参产物（tuned）' : '.env 配置'}</b>
            ，与线上检索实际使用的参数一致。
          </div>
        </>
      ),
    }

    const evalSlide: Slide = {
      key: 'evaluation',
      nav: '离线评估',
      eyebrow: '04 / 离线评估',
      title: '检索质量 · 离线测试集',
      lead: evaluation
        ? `最近一次评估：${evaluation.questions ?? '—'} 题测试集，模式 ${evaluation.mode ?? '—'}。以下为离线指标，非线上实时数据。`
        : '暂未找到评估结果文件。',
      body: evaluation ? (
        <>
          <div className="sc-metrics">
            <Metric
              value={fmtPct(evaluation.hit_rate)}
              label="检索命中率"
              hint="正确文档进入 Top-K 的比例"
              tone="proof"
            />
            <Metric
              value={fmtFixed(evaluation.mrr, 4)}
              label="MRR"
              hint="正确文档的排名倒数均值"
              tone="proof"
            />
            <Metric
              value={fmtPct(evaluation.coverage)}
              label="要点覆盖率"
              hint="回答覆盖参考答案要点的比例"
            />
            <Metric
              value={fmtPct(evaluation.similarity)}
              label="语义相似度"
              hint="回答与参考答案的向量相似度"
            />
            <Metric
              value={fmtFixed(evaluation.latency_s, 2)}
              unit="s"
              label="单题平均耗时"
              hint="含召回、精排与生成"
              tone="warn"
            />
          </div>
          <div className="sc-meta-strip">
            <span>评估时间 {fmtTime(evaluation.timestamp)}</span>
            <span>测试集 {evaluation.testset ?? '—'}</span>
            <span>模式 {evaluation.mode ?? '—'}</span>
            <span>
              召回 Top-{evaluation.recall_k ?? '—'} · 精排 Top-
              {evaluation.reranker_top_k ?? '—'}
            </span>
            <span>
              权重 BM25 {fmtFixed(evaluation.bm25_weight)} / 向量{' '}
              {fmtFixed(evaluation.vector_weight)}
            </span>
          </div>
        </>
      ) : (
        <div className="sc-empty">
          <Icon name="file" size="lg" />
          <span>未找到 <code>data/eval</code> 下的评估结果，跑一次 <code>eval_rag.py</code> 后即可展示。</span>
        </div>
      ),
    }

    const fb = feedback
    const runSlide: Slide = {
      key: 'usage',
      nav: '运行与反馈',
      eyebrow: '05 / 运行与反馈',
      title: '使用量与人工闭环',
      lead: '左为累计使用量，右为反馈汇总 —— 差评会连同用户的纠错文本进入反馈看板，供人工修订语料。',
      body: (
        <>
          <div className="sc-split">
            <div className="sc-panel">
              <div className="sc-panel-head">
                <Icon name="chat" size="sm" />
                累计使用量
              </div>
              <div className="sc-pairs">
                <div className="sc-pair">
                  <span className="sc-pair-n">{fmtInt(usage?.conversations)}</span>
                  <span className="sc-pair-l">会话数</span>
                </div>
                <div className="sc-pair">
                  <span className="sc-pair-n">{fmtInt(usage?.questions)}</span>
                  <span className="sc-pair-l">提问数</span>
                </div>
                <div className="sc-pair">
                  <span className="sc-pair-n">{fmtInt(usage?.answers)}</span>
                  <span className="sc-pair-l">回答数</span>
                </div>
                <div className="sc-pair is-proof">
                  <span className="sc-pair-n">{fmtInt(usage?.clarifications)}</span>
                  <span className="sc-pair-l">证据不足 · 触发追问</span>
                </div>
              </div>
            </div>

            <div className="sc-panel">
              <div className="sc-panel-head">
                <Icon name="feedback" size="sm" />
                反馈汇总
              </div>
              <div className="sc-pairs">
                <div className="sc-pair">
                  <span className="sc-pair-n">{fmtInt(fb?.total)}</span>
                  <span className="sc-pair-l">反馈总数</span>
                </div>
                <div className="sc-pair">
                  <span className="sc-pair-n sc-ok">{fmtInt(fb?.up)}</span>
                  <span className="sc-pair-l">标记准确</span>
                </div>
                <div className="sc-pair">
                  <span className="sc-pair-n sc-danger">{fmtInt(fb?.down)}</span>
                  <span className="sc-pair-l">标记不准确</span>
                </div>
                <div className="sc-pair">
                  <span className="sc-pair-n">{fmtInt(fb?.with_correction)}</span>
                  <span className="sc-pair-l">附人工纠错</span>
                </div>
              </div>
            </div>
          </div>
          <div className="sc-note">
            「触发追问」是<b>证据不足时不硬答</b>这一机制真实生效的次数。
            {fb && fb.total > 0 && fb.total < 30 && (
              <span className="sc-warn-inline">
                <Icon name="alert" size="sm" />
                反馈样本仅 {fb.total} 条，比例类指标暂不具统计意义
              </span>
            )}
          </div>
        </>
      ),
    }

    return [corpusSlide, deptSlide, retrievalSlide, evalSlide, runSlide]
  }, [data, corpus, evaluation, usage, feedback])

  // ---- 轮播：定时推进；进度条用 CSS 动画，key 变化即重启动画 ----
  const slideCount = slides.length
  useEffect(() => {
    if (paused || slideCount <= 1) return
    const t = setTimeout(() => setIndex((i) => (i + 1) % slideCount), SLIDE_MS)
    return () => clearTimeout(t)
  }, [index, paused, slideCount])

  const go = useCallback(
    (next: number) => setIndex(((next % slideCount) + slideCount) % slideCount),
    [slideCount],
  )

  const goRef = useRef(go)
  goRef.current = go

  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === 'ArrowRight') goRef.current(index + 1)
      else if (e.key === 'ArrowLeft') goRef.current(index - 1)
      else if (e.key === ' ') {
        e.preventDefault()
        setPaused((p) => !p)
      } else if (e.key === 'f' || e.key === 'F') {
        if (document.fullscreenElement) document.exitFullscreen().catch(() => {})
        else document.documentElement.requestFullscreen().catch(() => {})
      } else if (e.key === 'Escape' && !document.fullscreenElement) {
        onExit()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [index, onExit])

  function toggleTheme() {
    const next =
      document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark'
    document.documentElement.dataset.theme = next
    localStorage.setItem('medical-rag.theme', next)
  }

  if (!state) {
    return (
      <div className="screen">
        <div className="sc-loading">
          <Icon name="pulse" size="lg" />
          正在加载大屏数据…
        </div>
      </div>
    )
  }

  return (
    <div className="screen" role="region" aria-label="数据大屏">
      <header className="sc-head">
        <span className="sc-mark">
          <Icon name="pulse" />
        </span>
        <div className="sc-brand">
          <strong>医疗智能问答 · 运行数据</strong>
          <span>
            {state.live ? '接口实时数据' : '兜底快照'} · 数据时间{' '}
            {fmtTime(data?.generated_at)}
          </span>
        </div>

        <span className={`sc-src${state.live ? ' is-live' : ''}`}>
          <span className="sc-src-dot" />
          {state.live ? '实时' : '快照'}
        </span>

        <div className="sc-actions">
          <button
            className="sc-btn"
            type="button"
            title="立即刷新"
            aria-label="立即刷新"
            onClick={load}
          >
            <Icon name="refresh" />
          </button>
          <button
            className="sc-btn"
            type="button"
            title="切换主题"
            aria-label="切换主题"
            onClick={toggleTheme}
          >
            <Icon name="sun" />
          </button>
          <button
            className="sc-btn"
            type="button"
            title={isFull ? '退出全屏' : '进入全屏'}
            aria-label={isFull ? '退出全屏' : '进入全屏'}
            onClick={() => {
              if (document.fullscreenElement) document.exitFullscreen().catch(() => {})
              else document.documentElement.requestFullscreen().catch(() => {})
            }}
          >
            <Icon name={isFull ? 'shrink' : 'expand'} />
          </button>
          <button className="sc-btn sc-btn-text" type="button" onClick={onExit}>
            <Icon name="x" size="sm" />
            返回工作台
          </button>
        </div>
      </header>

      <main className="sc-body">
        {slides.map((s, i) => (
          <section
            key={s.key}
            className={`sc-slide${i === index ? ' is-on' : ''}`}
            aria-hidden={i !== index}
          >
            <div className="sc-slide-head">
              <span className="sc-eyebrow">{s.eyebrow}</span>
              <h1>{s.title}</h1>
              <p>{s.lead}</p>
            </div>
            <div className="sc-slide-body">{s.body}</div>
          </section>
        ))}
      </main>

      <footer className="sc-foot">
        <button
          className="sc-btn sc-btn-text"
          type="button"
          onClick={() => setPaused((p) => !p)}
          aria-label={paused ? '继续轮播' : '暂停轮播'}
        >
          <Icon name={paused ? 'chev' : 'stop'} size="sm" />
          {paused ? '继续轮播' : '暂停轮播'}
        </button>

        <div className="sc-progress" role="tablist" aria-label="大屏分屏">
          {slides.map((s, i) => (
            <button
              key={s.key}
              className={`sc-seg${i === index ? ' is-on' : ''}`}
              type="button"
              role="tab"
              aria-selected={i === index}
              title={s.nav}
              onClick={() => go(i)}
            >
              <span className="sc-seg-label">{s.nav}</span>
              <span className="sc-seg-track">
                <span
                  key={`${s.key}-${index}-${paused}`}
                  className="sc-seg-fill"
                  style={{
                    animationDuration: `${SLIDE_MS}ms`,
                    animationPlayState: paused ? 'paused' : 'running',
                  }}
                />
              </span>
            </button>
          ))}
        </div>

        <span className="sc-count">
          {index + 1} / {slides.length}
        </span>
      </footer>

      {fsHint && !isFull && (
        <div className="sc-hint">
          <Icon name="expand" size="sm" />
          点击任意处进入全屏 · ← → 切屏 · 空格暂停
        </div>
      )}
    </div>
  )
}