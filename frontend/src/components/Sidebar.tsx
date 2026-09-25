import { useState } from 'react'
import type { Conversation } from '../api/chat'

interface SidebarProps {
  conversations: Conversation[]
  activeId: number | null
  onSelect: (id: number) => void
  onCreate: () => void
  onDelete: (id: number) => void
}

export default function Sidebar({
  conversations,
  activeId,
  onSelect,
  onCreate,
  onDelete,
}: SidebarProps) {
  const [hoverId, setHoverId] = useState<number | null>(null)

  return (
    <div className="sidebar">
      <div className="sidebar-header">
        <button className="new-chat-btn" onClick={onCreate}>
          + 新对话
        </button>
      </div>
      <div className="sidebar-list">
        {conversations.length === 0 && (
          <div className="sidebar-empty">暂无对话</div>
        )}
        {conversations.map((c) => (
          <div
            key={c.id}
            className={`sidebar-item ${activeId === c.id ? 'active' : ''}`}
            onClick={() => onSelect(c.id)}
            onMouseEnter={() => setHoverId(c.id)}
            onMouseLeave={() => setHoverId(null)}
          >
            <div className="sidebar-item-title" title={c.title}>
              {c.title || '新对话'}
            </div>
            {hoverId === c.id && (
              <button
                className="sidebar-item-delete"
                onClick={(e) => {
                  e.stopPropagation()
                  if (confirm('确定删除这个对话吗？')) {
                    onDelete(c.id)
                  }
                }}
                title="删除"
              >
                ×
              </button>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}
