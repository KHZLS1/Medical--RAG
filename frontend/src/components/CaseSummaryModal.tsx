import { useMemo, useEffect } from 'react'
import Icon from './Icon'
import CopyButton from './CopyButton'
import type { Source } from '../api/chat'

export interface SummaryMessage {
  role: 'user' | 'assistant'
  content: string
  sources?: Source[]
  clarify?: unknown
}

interface CaseSummaryModalProps {
  conversationTitle: string
  messages: SummaryMessage[]
  onClose: () => void
}

/** 纯确认/寒暄词表，用于提取主诉时剔除无效口语 */
const PURE_ACKS = new Set([
  '好', '好的', '好滴', '好哒', '行', '可以', '嗯', '嗯嗯', '对', '对的',
  '是的', '收到', '明白了', '知道了', '谢谢', '感谢', '辛苦了', '不了', '算了',
])

function cleanText(t: string): string {
  return t.replace(/[*#>`~_]/g, '').trim()
}

export default function CaseSummaryModal({
  conversationTitle,
  messages,
  onClose,
}: CaseSummaryModalProps) {
  // 键盘 Esc 关闭
  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [onClose])

  // 结构化提炼 SOAP 元素
  const summary = useMemo(() => {
    // 1. S (Subjective / 主诉)
    const userComplaints = messages
      .filter((m) => m.role === 'user')
      .map((m) => m.content.trim())
      .filter((t) => t && !PURE_ACKS.has(t))

    const chiefComplaint = userComplaints.length > 0
      ? userComplaints.join('；')
      : conversationTitle || '患者在线咨询'

    // 2. O (Objective / 检索依据)
    const allSources: Source[] = []
    const seenTitles = new Set<string>()
    for (const m of messages) {
      if (m.role === 'assistant' && m.sources) {
        for (const s of m.sources) {
          if (!seenTitles.has(s.title)) {
            seenTitles.add(s.title)
            allSources.push(s)
          }
        }
      }
    }

    // 3. A (Assessment / AI初步分析) & P (Plan / 建议措施)
    const assistantAnswers = messages
      .filter((m) => m.role === 'assistant' && m.content && !m.clarify)
      .map((m) => m.content)

    // 提取可能的病因或分析段落
    const combinedAnswer = assistantAnswers.join('\n\n')

    // 建议挂号科室推测
    const departments = Array.from(
      new Set(allSources.map((s) => s.department).filter(Boolean)),
    )
    const recommendedDept = departments.length > 0 ? departments.join(' / ') : '综合内科 / 全科医学科'

    // 打印当前时间
    const generatedTime = new Date().toLocaleString('zh-CN', {
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
    })

    return {
      chiefComplaint,
      allSources,
      recommendedDept,
      generatedTime,
      combinedAnswer,
    }
  }, [messages, conversationTitle])

  // 组装格式化 Markdown / 文本
  const fullReportText = useMemo(() => {
    return [
      `# 医疗智能问答 · 门诊预问诊小结`,
      `**咨询主题**：${conversationTitle || '健康咨询'}`,
      `**生成时间**：${summary.generatedTime}`,
      `**建议挂号科室**：${summary.recommendedDept}`,
      ``,
      `---`,
      ``,
      `### 一、 【S】主诉与病情自述`,
      `${summary.chiefComplaint}`,
      ``,
      `### 二、 【A】AI 初步健康分析要点`,
      summary.combinedAnswer
        ? cleanText(summary.combinedAnswer.slice(0, 800)) + (summary.combinedAnswer.length > 800 ? '…' : '')
        : '（暂无详细病情分析）',
      ``,
      `### 三、 【P】就医建议与注意事项`,
      `- **推荐就诊科室**：${summary.recommendedDept}`,
      `- **就诊提醒**：请携带既往病历、近期体检化验单及正在服用的药物包装就医。`,
      `- **安全红线**：如出现剧烈胸痛、呼吸急促困难、大出血或意识不清，请立即就近就医或拨打 120 急救。`,
      ``,
      `### 四、 【O】参考医学资料来源 (${summary.allSources.length} 篇)`,
      summary.allSources.length > 0
        ? summary.allSources.map((s, idx) => `${idx + 1}. [${s.department || '医学'}] ${s.title}`).join('\n')
        : '（无外部检索资料）',
      ``,
      `---`,
      `⚠️ 免责声明：本小结由 AI 辅助根据对话信息生成，仅供患者就诊时与执业医师沟通参考，不能替代医师执业诊断与处方。`,
    ].join('\n')
  }, [conversationTitle, summary])

  function handlePrint() {
    window.print()
  }

  return (
    <aside
      className="sheet case-summary-sheet"
      role="dialog"
      aria-modal="true"
      aria-label="门诊预问诊小结"
    >
      <div className="sheet-head">
        <Icon name="file" size="sm" />
        <h2>门诊预问诊小结</h2>
        <button className="btn btn-icon" type="button" aria-label="关闭" onClick={onClose}>
          <Icon name="x" size="sm" />
        </button>
      </div>

      <div className="sheet-body">
        <div className="summary-card-preview">
          <div className="summary-report-header">
            <div className="summary-hospital-title">医疗智能问答系统 · 预问诊小结</div>
            <div className="summary-meta-row">
              <span><b>会话主题：</b>{conversationTitle || '医疗咨询'}</span>
              <span><b>生成时间：</b>{summary.generatedTime}</span>
            </div>
            <div className="summary-badge-row">
              <span className="summary-badge">建议首诊科室：{summary.recommendedDept}</span>
              <span className="summary-badge summary-badge-soft">参考资料：{summary.allSources.length} 篇</span>
            </div>
          </div>

          <div className="summary-section">
            <div className="summary-section-label">
              <span className="soap-tag">S</span>
              <span>主诉与病情自述（Subjective）</span>
            </div>
            <div className="summary-section-content">{summary.chiefComplaint}</div>
          </div>

          <div className="summary-section">
            <div className="summary-section-label">
              <span className="soap-tag">A</span>
              <span>AI 初步健康分析（Assessment）</span>
            </div>
            <div className="summary-section-content summary-content-lead">
              {summary.combinedAnswer ? (
                summary.combinedAnswer.slice(0, 450) + (summary.combinedAnswer.length > 450 ? '……' : '')
              ) : (
                '（暂无完整问诊分析）'
              )}
            </div>
          </div>

          <div className="summary-section">
            <div className="summary-section-label">
              <span className="soap-tag">P</span>
              <span>建议措施与诊疗计划（Plan）</span>
            </div>
            <ul className="summary-plan-list">
              <li>
                <strong>就诊科室</strong>：建议优先前往 <b>{summary.recommendedDept}</b> 挂号面诊。
              </li>
              <li>
                <strong>资料准备</strong>：建议携带既往病历本、近 3 个月化验单及正在服用的药物包装。
              </li>
              <li>
                <strong>用药警示</strong>：本系统不提供具体药物剂量，如需用药请务必严格遵医嘱。
              </li>
              <li>
                <strong>危急提示</strong>：若伴随剧烈胸闷胸痛、呼吸急促、咯血或神志不清，请立即拨打 120 呼叫急救。
              </li>
            </ul>
          </div>

          {summary.allSources.length > 0 && (
            <div className="summary-section">
              <div className="summary-section-label">
                <span className="soap-tag">O</span>
                <span>参考医学依据（Objective References）</span>
              </div>
              <div className="summary-sources-tags">
                {summary.allSources.map((s, idx) => (
                  <span key={idx} className="summary-source-pill" title={s.source}>
                    [{idx + 1}] {s.department ? `${s.department} · ` : ''}{s.title}
                  </span>
                ))}
              </div>
            </div>
          )}

          <div className="summary-disclaimer">
            <Icon name="alert" size="sm" />
            <span>
              本小结由 AI 助手根据对话内容提炼，<b>仅供线下就诊沟通参考</b>，不能替代执业医师诊断证明。
            </span>
          </div>
        </div>

        <div className="summary-actions-bar">
          <CopyButton
            text={fullReportText}
            label="复制完整病历小结 (Markdown)"
            className="btn btn-accent"
          />
          <button className="btn" type="button" onClick={handlePrint}>
            <Icon name="panel" size="sm" />
            打印 / 另存为 PDF
          </button>
        </div>
      </div>
    </aside>
  )
}
