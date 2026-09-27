import { useEffect } from 'react'
import Icon from './Icon'

/**
 * 首次进入的海报页。
 *
 * 为什么是"仅首次"：这是个工具，不是官网。每次进来都挡一屏，日常用户几天就会烦，
 * 所以由 App 用 localStorage 记标记，看过就不再出现；想重看从左侧图标栏的
 * 「使用说明」随时能开。
 *
 * 内容取向（面向内部使用者）：不讲"AI 医疗问答"这种泛话术，只讲三件他们真正
 * 需要知道的事 —— 回答为什么可信、库里的东西不够时系统会怎么处理、点哪里能核对。
 * 因此主视觉用检索链路，而不是产品口号；青绿贯穿到最后一环，和正文引用角标同色。
 *
 * 数字口径：605,274 / 232 由 backend/data/by_department 实际统计得出；
 * 命中率取自 backend/data/eval 里 2026-09-26 那次评估（50 题，Top-20，精排 Top-5），
 * 是离线指标 —— 页脚必须写明来源，否则内部同事会当成线上实时数据。
 */
const CHAIN = [
  { n: '01', icon: 'user', t: '你的问题', d: '口语描述即可，不必是标准医学术语' },
  { n: '02', icon: 'search', t: '改写补全', d: '多轮里省略的主语、指代词，结合上文补齐' },
  { n: '03', icon: 'route', t: '混合召回', d: 'BM25 稀疏 + 向量稠密，各召回 20 条' },
  { n: '04', icon: 'panel', t: '精排', d: 'Cross-Encoder 重排，留 Top-5 作为依据' },
  { n: '05', icon: 'link', t: '引用作答', d: '严格限于资料范围，逐条标注 [1][2] 编号', proof: true },
] as const

const CAPS = [
  {
    icon: 'link',
    t: '引用可核对',
    d: '正文里的编号可点，右侧证据栏展开该条原文、科室与来源路径。资料里没有的，不会编。',
  },
  {
    icon: 'chat',
    t: '证据不足会追问',
    d: '库中检索不到依据时不硬答，先向你确认症状、部位或时长，补充后再重新检索。',
  },
  {
    icon: 'route',
    t: '多轮记得上下文',
    d: '「那要吃什么药」这类省略主语的追问，会自动结合上文补全，而不是裸着去检索。',
  },
  {
    icon: 'feedback',
    t: '反馈进看板',
    d: '答得不对可点「不准确」并写下你认为正确的回答，会进入反馈看板，供人工修订语料。',
  },
] as const

const STATS = [
  { n: '605,274', l: '清洗后入库语料（条）' },
  { n: '232', l: '覆盖科室与病种' },
  { n: '70%', l: '检索命中率 · 离线评估' },
] as const

export default function Intro({ onClose }: { onClose: () => void }) {
  // 海报是全屏遮罩，键盘用户必须有出口，否则 Tab 会被困在里面
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  return (
    <div className="intro" role="dialog" aria-modal="true" aria-label="使用说明">
      <div className="intro-panel">
        <div className="intro-top">
          <span className="intro-mark">
            <Icon name="pulse" />
          </span>
          <span className="intro-brand">
            <strong>医疗智能问答</strong>
            <span>循证工作台 · 内部试用</span>
          </span>
          <button className="btn intro-skip" type="button" onClick={onClose}>
            <Icon name="x" size="sm" />
            跳过
          </button>
        </div>

        <div className="intro-hero">
          <div className="intro-eyebrow">每条回答，都能点开核对原文</div>
          <h1>先给依据，再给结论</h1>
          <p>
            回答严格限定在检索到的医学资料范围内，并逐条标注 [1][2] 引用编号。
            点编号即可在右侧证据栏展开原文与科室来源，信不信由你自己判断 ——
            而不是让模型自说自话。
          </p>
        </div>

        <div className="intro-chain" aria-label="检索链路">
          {CHAIN.map((s) => (
            <div key={s.n} className={`step${'proof' in s && s.proof ? ' is-proof' : ''}`}>
              <span className="step-top">
                <span className="step-n">{s.n}</span>
                <Icon name={s.icon} size="sm" />
              </span>
              <span className="step-t">{s.t}</span>
              <span className="step-d">{s.d}</span>
            </div>
          ))}
        </div>

        <div className="intro-caps">
          {CAPS.map((c) => (
            <div key={c.t} className="cap">
              <Icon name={c.icon} size="sm" />
              <span className="cap-body">
                <span className="cap-t">{c.t}</span>
                <span className="cap-d">{c.d}</span>
              </span>
            </div>
          ))}
        </div>

        <div className="intro-stats">
          {STATS.map((s) => (
            <div key={s.l} className="stat">
              <div className="stat-n">{s.n}</div>
              <div className="stat-l">{s.l}</div>
            </div>
          ))}
        </div>

        <div className="intro-foot">
          <p className="intro-note">
            命中率取自 2026-09-26 的离线评估（50 题测试集，Top-20 召回，BM25 0.3 / 向量 0.7，精排 Top-5），
            非线上实时指标。回答仅基于公开医学资料，仅供参考，<b>不能替代执业医师诊断</b>；
            涉及急症请立即拨打 120。
          </p>
          <button className="btn btn-accent intro-cta" type="button" autoFocus onClick={onClose}>
            进入工作台
          </button>
        </div>
      </div>
    </div>
  )
}