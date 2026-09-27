-- ============================================================
-- feedback 表幂等约束迁移
-- 背景：反馈看板此前允许同一条消息重复提交反馈，产生重复行。
-- 现在业务改为「一条消息只保留一条反馈」，先清理历史重复，再加唯一约束。
-- create_all 不会为已存在的表加约束，需手动执行本脚本（MySQL 8）。
-- 用法：mysql -u<user> -p <database> < migrate_feedback_unique.sql
-- ============================================================

-- 0) 清理历史重复：每个 message_id 只保留 id 最大（最新）的一条反馈，其余删除
DELETE f FROM feedback f
LEFT JOIN (
    SELECT MAX(id) AS id FROM feedback GROUP BY message_id
) keep ON keep.id = f.id
WHERE keep.id IS NULL;

-- 1) 为 stats 按 thumbs 过滤 + 按时间倒序分页的查询建索引
CREATE INDEX idx_feedback_thumbs_time ON feedback (thumbs, created_at);

-- 2) 唯一约束：同一 message_id 只允许一条反馈（幂等的库层兜底）
ALTER TABLE feedback
  ADD CONSTRAINT uq_feedback_message_id UNIQUE (message_id);