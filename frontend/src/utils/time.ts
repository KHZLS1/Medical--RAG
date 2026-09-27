/**
 * 时间显示口径。
 *
 * 为什么单独抽一个模块：消息区、证据栏、问诊小结都要回答"这条是什么时候产生的"，
 * 各写一份必然出现"有的带秒、有的不带、跨天还不一样"的不一致。口径统一放这里。
 */

/** 后端给的是不带时区的本地时间串（`2026-09-27T19:41:26`），不是 UTC */
const NAIVE_LOCAL = /^(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2}):(\d{2})/
const HAS_ZONE = /(?:[Zz]|[+-]\d{2}:?\d{2})$/

/**
 * 解析后端时间串。
 *
 * 为什么要手写而不用 `new Date(iso)`：ES 规范里"不带时区的 date-time"按本地时间
 * 解释、"只有日期"按 UTC 解释 —— 这条规则很容易在换后端字段（比如某天改成
 * `datetime.now(timezone.utc)` 带上 Z）之后默默把时间显示偏 8 小时。
 * 所以这里显式区分：没有时区标记就按本地时间构造，带了时区才交给 Date 自己算。
 */
function toDate(iso: string): Date | null {
  const m = NAIVE_LOCAL.exec(iso)
  const d = m && !HAS_ZONE.test(iso)
    ? new Date(+m[1], +m[2] - 1, +m[3], +m[4], +m[5], +m[6])
    : new Date(iso)
  return Number.isNaN(d.getTime()) ? null : d
}

const pad = (n: number) => String(n).padStart(2, '0')

/**
 * 消息时间（紧凑口径）：当天只给 `时:分`，跨天补 `月日`，跨年再补年份。
 * 与左侧会话列表的时间口径一致 —— 同一屏里两种写法会让人以为数据来源不同。
 */
export function formatStamp(iso?: string | null): string {
  if (!iso) return ''
  const d = toDate(iso)
  if (!d) return ''
  const now = new Date()
  const hm = `${pad(d.getHours())}:${pad(d.getMinutes())}`
  if (d.toDateString() === now.toDateString()) return hm
  if (d.getFullYear() === now.getFullYear()) {
    return `${d.getMonth() + 1}月${d.getDate()}日 ${hm}`
  }
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${hm}`
}

/**
 * 完整时间戳 `YYYY-MM-DD HH:mm:ss`，给 `title` 悬浮提示用。
 *
 * 界面上只显示到分钟（够用且不吵），但"这一条到底哪一秒到的"在排查
 * 流式时序、对齐后端日志时是必需的 —— 悬浮就能看到，不用去翻库。
 */
export function formatFullStamp(iso?: string | null): string {
  if (!iso) return ''
  const d = toDate(iso)
  if (!d) return ''
  return (
    `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ` +
    `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`
  )
}

/**
 * 两个时间点之间的耗时，给"这条回答等了多久"用。
 * 不足 1 秒给 `1 秒内`（不显示 0 秒，读起来像没花时间）；超过 1 分钟给 `1分23秒`。
 */
export function formatDuration(fromIso?: string | null, toIso?: string | null): string {
  if (!fromIso || !toIso) return ''
  const a = toDate(fromIso)
  const b = toDate(toIso)
  if (!a || !b) return ''
  const sec = Math.round((b.getTime() - a.getTime()) / 1000)
  if (sec < 0) return ''
  if (sec < 1) return '1 秒内'
  if (sec < 60) return `${sec} 秒`
  return `${Math.floor(sec / 60)}分${sec % 60}秒`
}
