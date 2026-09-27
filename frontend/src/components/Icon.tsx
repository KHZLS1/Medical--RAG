/**
 * 统一描边图标集。
 *
 * 为什么替掉 emoji：emoji 的字形由系统字体决定，Windows / macOS / 安卓
 * 三处大小、基线、色彩全不一样，同一排按钮在不同机器上会歪得各不相同；
 * 而且 emoji 自带颜色，和"临床蓝 / 证据青绿"这套克制的配色打架。
 * 这里全部改成 24 格描边路径，stroke 继承 currentColor，颜色由 CSS 令牌决定。
 */
const ICONS = {
  pulse: <path d="M3 12h3.6l2.1-5.4 3.2 10.8L14.4 12H21" />,
  plus: <path d="M12 5v14M5 12h14" />,
  chat: (
    <path d="M20.5 12.2a7.7 7.7 0 0 1-7.7 7.7H8.4L4 22.3l1.2-4.3a7.7 7.7 0 1 1 15.3-5.8Z" />
  ),
  file: (
    <>
      <path d="M13.5 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8.5L13.5 3Z" />
      <path d="M13.5 3v5.5H19" />
    </>
  ),
  feedback: (
    <>
      <path d="M20.5 14.5a2 2 0 0 1-2 2H8.4L4 21V6.5a2 2 0 0 1 2-2h12.5a2 2 0 0 1 2 2Z" />
      <path d="M9 10h6M9 13.5h3.5" />
    </>
  ),
  sun: (
    <>
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2v2.4M12 19.6V22M4.2 4.2l1.7 1.7M18.1 18.1l1.7 1.7M2 12h2.4M19.6 12H22M4.2 19.8l1.7-1.7M18.1 5.9l1.7-1.7" />
    </>
  ),
  moon: <path d="M20.8 13.3A8.7 8.7 0 1 1 10.7 3.2a6.8 6.8 0 0 0 10.1 10.1Z" />,
  refresh: (
    <>
      <path d="M20 12a8 8 0 0 0-13.7-5.6L4 8.6" />
      <path d="M4 5v4h4" />
      <path d="M4 12a8 8 0 0 0 13.7 5.6L20 15.4" />
      <path d="M20 19v-4h-4" />
    </>
  ),
  panel: (
    <>
      <rect x="3" y="4" width="18" height="16" rx="2.5" />
      <path d="M14.5 4v16" />
    </>
  ),
  send: (
    <>
      <path d="M21.5 2.5 10.8 13.2" />
      <path d="M21.5 2.5 14.6 21.5l-3.8-8.3-8.3-3.8 19-7Z" />
    </>
  ),
  stop: <rect x="6.5" y="6.5" width="11" height="11" rx="2" />,
  copy: (
    <>
      <rect x="8.5" y="8.5" width="11" height="11" rx="2" />
      <path d="M15.5 5.5V5a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v7.5a2 2 0 0 0 2 2h.5" />
    </>
  ),
  check: <path d="M4.5 12.5 9.5 17.5 19.5 6.5" />,
  up: (
    <>
      <path d="M7 21.5V10.2l3.9-8.2a2.4 2.4 0 0 1 2.3 2.4v4.3h4.6a2.2 2.2 0 0 1 2.2 2.6l-1.3 7.1a2.4 2.4 0 0 1-2.4 2H7Z" />
      <path d="M3 21.5h4V10.2H3Z" />
    </>
  ),
  down: (
    <>
      <path d="M17 2.5v11.3l-3.9 8.2a2.4 2.4 0 0 1-2.3-2.4v-4.3H6.2a2.2 2.2 0 0 1-2.2-2.6l1.3-7.1a2.4 2.4 0 0 1 2.4-2H17Z" />
      <path d="M21 2.5h-4v11.3h4Z" />
    </>
  ),
  alert: (
    <>
      <path d="M12 3.2 21.3 20H2.7L12 3.2Z" />
      <path d="M12 9.5v4.2M12 17.1h.01" />
    </>
  ),
  chev: <path d="M6 9.5 12 15.5 18 9.5" />,
  x: <path d="M6 6l12 12M18 6 6 18" />,
  search: (
    <>
      <circle cx="11" cy="11" r="6.8" />
      <path d="M20 20l-4.1-4.1" />
    </>
  ),
  trash: (
    <>
      <path d="M4 7h16M9.5 7V4.8h5V7" />
      <path d="M6.4 7l.9 12.4h9.4L17.6 7" />
    </>
  ),
  upload: (
    <>
      <path d="M12 16.5V4.2" />
      <path d="M7.4 8.8 12 4.2l4.6 4.6" />
      <path d="M4 20h16" />
    </>
  ),
  user: (
    <>
      <circle cx="12" cy="8.5" r="3.8" />
      <path d="M4.8 20.2a7.2 7.2 0 0 1 14.4 0" />
    </>
  ),
  stethoscope: (
    <>
      <path d="M6 3v5.2a4.4 4.4 0 0 0 8.8 0V3" />
      <path d="M10.4 12.6v2.6a5 5 0 0 0 5 5h.6a4 4 0 0 0 4-4v-1.6" />
      <circle cx="20" cy="12.4" r="1.6" />
    </>
  ),
  link: (
    <>
      <path d="M10 13.5a3.6 3.6 0 0 0 5.1 0l3.3-3.3a3.6 3.6 0 0 0-5.1-5.1L11.6 6.8" />
      <path d="M14 10.5a3.6 3.6 0 0 0-5.1 0l-3.3 3.3a3.6 3.6 0 0 0 5.1 5.1l1.7-1.7" />
    </>
  ),
  route: (
    <>
      <circle cx="6" cy="6" r="2.6" />
      <circle cx="18" cy="18" r="2.6" />
      <path d="M8.6 6h5.4a4 4 0 0 1 0 8H9.4a4 4 0 0 0 0 8h6" />
    </>
  ),
  info: (
    <>
      <circle cx="12" cy="12" r="8.6" />
      <path d="M12 11v5.4M12 7.9h.01" />
    </>
  ),
  /* 全屏进出：四角朝外的折线，比"方框+箭头"在小尺寸下更清楚 */
  expand: <path d="M9 3.5H3.5V9M15 3.5h5.5V9M15 20.5h5.5V15M9 20.5H3.5V15" />,
  shrink: <path d="M3.5 9H9V3.5M20.5 9H15V3.5M20.5 15H15v5.5M3.5 15H9v5.5" />,
} as const

export type IconName = keyof typeof ICONS

const SIZE_CLASS = { sm: 'icon-sm', md: 'icon', lg: 'icon-lg' } as const

export default function Icon({
  name,
  size = 'md',
  className = '',
}: {
  name: IconName
  size?: keyof typeof SIZE_CLASS
  className?: string
}) {
  return (
    <svg
      className={`${SIZE_CLASS[size]}${className ? ` ${className}` : ''}`}
      viewBox="0 0 24 24"
      aria-hidden="true"
      focusable="false"
    >
      {ICONS[name]}
    </svg>
  )
}