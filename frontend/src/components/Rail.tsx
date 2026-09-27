import type { Conversation } from '../api/chat'
import Icon from './Icon'

interface RailProps {
  conversations: Conversation[]
  activeId: number | null
  expanded: boolean
  onToggleExpanded: () => void
  onSelect: (id: number) => void
  onCreate: () => void
  onDelete: (id: number) => void
  theme: 'light' | 'dark'
  onToggleTheme: () => void
  onOpenDocs: () => void
  onOpenFeedback: () => void
  onOpenIntro: () => void
  onOpenScreen: () => void
}

/** 会话时间：当天只给时分，跨天给月-日，跨年再补年份。 */
function shortTime(iso: string): string {
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return ''
  const now = new Date()
  const p = (n: number) => String(n).padStart(2, '0')
  if (d.toDateString() === now.toDateString()) return `${p(d.getHours())}:${p(d.getMinutes())}`
  if (d.getFullYear() === now.getFullYear()) return `${d.getMonth() + 1}月${d.getDate()}日`
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}`
}

export default function Rail({
  conversations,
  activeId,
  expanded,
  onToggleExpanded,
  onSelect,
  onCreate,
  onDelete,
  theme,
  onToggleTheme,
  onOpenDocs,
  onOpenFeedback,
  onOpenIntro,
  onOpenScreen,
}: RailProps) {
  return (
    <aside className="rail" aria-label="主导航">
      <div className="rail-brand">
        <span className="rail-mark">
          <Icon name="pulse" />
        </span>
        <span className="rail-brand-text">
          <strong>医疗智能问答</strong>
          <span>循证工作台</span>
        </span>
      </div>

      <button className="rail-btn rail-primary" type="button" title="新建对话" onClick={onCreate}>
        <Icon name="plus" />
        <span className="rail-label">新建对话</span>
      </button>

      <button
        className={`rail-btn${expanded ? ' is-on' : ''}`}
        type="button"
        title="会话列表"
        aria-expanded={expanded}
        onClick={onToggleExpanded}
      >
        <Icon name="chat" />
        <span className="rail-label">会话列表</span>
      </button>

      <div className="rail-list" aria-label="历史会话">
        <div className="rail-list-label">历史会话</div>
        {conversations.length === 0 && <div className="rail-list-empty">暂无对话</div>}
        {conversations.map((c) => (
          <div
            key={c.id}
            className="conv"
            role="button"
            tabIndex={0}
            aria-current={activeId === c.id}
            onClick={() => onSelect(c.id)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault()
                onSelect(c.id)
              }
            }}
          >
            <span className="conv-body">
              <span className="conv-title" title={c.title}>
                {c.title || '新对话'}
              </span>
              <span className="conv-time">{shortTime(c.updated_at || c.created_at)}</span>
            </span>
            <button
              className="conv-del"
              type="button"
              aria-label="删除会话"
              title="删除"
              onClick={(e) => {
                e.stopPropagation()
                if (confirm(`确定删除「${c.title || '新对话'}」吗？`)) onDelete(c.id)
              }}
            >
              <Icon name="trash" size="sm" />
            </button>
          </div>
        ))}
      </div>

      <div className="rail-spacer" />

      <button className="rail-btn" type="button" title="使用说明" onClick={onOpenIntro}>
        <Icon name="info" />
        <span className="rail-label">使用说明</span>
      </button>
      <button className="rail-btn" type="button" title="数据大屏" onClick={onOpenScreen}>
        <Icon name="panel" />
        <span className="rail-label">数据大屏</span>
      </button>
      <button className="rail-btn" type="button" title="文档管理" onClick={onOpenDocs}>
        <Icon name="file" />
        <span className="rail-label">文档管理</span>
      </button>
      <button className="rail-btn" type="button" title="反馈看板" onClick={onOpenFeedback}>
        <Icon name="feedback" />
        <span className="rail-label">反馈看板</span>
      </button>
      <button
        className="rail-btn"
        type="button"
        title={theme === 'light' ? '切换到深色主题' : '切换到浅色主题'}
        onClick={onToggleTheme}
      >
        <Icon name={theme === 'light' ? 'moon' : 'sun'} />
        <span className="rail-label">切换主题</span>
      </button>
    </aside>
  )
}