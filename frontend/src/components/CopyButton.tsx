import { useRef, useState } from 'react'
import Icon from './Icon'

/**
 * 复制按钮：一键把纯文本写入剪贴板，1.6s 内显示"已复制"。
 *
 * 为什么复制的是**原始字符串**而不是 DOM 文本
 * -----------------------------------------
 * 回答正文里的 `[n]` 被渲染成了按钮元素，直接取 innerText
 * 容易把结构搅乱；复制原文能保留 `[n]` 编号，客户粘到别处引用仍然可读。
 *
 * 剪贴板降级
 * ---------
 * `navigator.clipboard` 只在安全上下文（https / localhost）可用 —— 局域网
 * 用 http 访问时它是 undefined，所以必须保留 execCommand 兜底，否则
 * "复制没反应"会出现在最容易被忽略的部署形态上。
 */
async function writeClipboard(text: string): Promise<boolean> {
  if (navigator.clipboard && window.isSecureContext) {
    try {
      await navigator.clipboard.writeText(text)
      return true
    } catch {
      // 权限被拒或非用户手势触发 → 落到下面的兜底
    }
  }
  try {
    const ta = document.createElement('textarea')
    ta.value = text
    ta.setAttribute('readonly', '')
    ta.style.position = 'fixed'
    ta.style.top = '-9999px'
    document.body.appendChild(ta)
    ta.select()
    const ok = document.execCommand('copy')
    document.body.removeChild(ta)
    return ok
  } catch {
    return false
  }
}

export default function CopyButton({
  text,
  label = '复制',
  className = '',
  title,
}: {
  /** 要复制的纯文本。为空时不渲染按钮 */
  text: string
  label?: string
  className?: string
  title?: string
}) {
  const [copied, setCopied] = useState(false)
  const timerRef = useRef<number | null>(null)

  if (!text) return null

  async function copy() {
    const ok = await writeClipboard(text)
    if (!ok) return
    setCopied(true)
    if (timerRef.current) window.clearTimeout(timerRef.current)
    timerRef.current = window.setTimeout(() => setCopied(false), 1600)
  }

  return (
    <button
      type="button"
      className={`copy-btn${copied ? ' copied' : ''}${className ? ` ${className}` : ''}`}
      onClick={copy}
      title={title || label}
    >
      <Icon name={copied ? 'check' : 'copy'} size="sm" />
      {copied ? '已复制' : label}
    </button>
  )
}