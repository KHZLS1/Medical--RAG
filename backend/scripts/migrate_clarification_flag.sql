-- ============================================================
-- chat_messages 加「追问话术」标记（阶段二）
--
-- 背景：证据不足时后端 interrupt 追问，追问话术也作为一条助手消息落库
-- （见 main.py 的 clarification_request 分支）。但表里没有字段区分它和
-- 普通回答，前端刷新后只能把它渲染成普通回答 —— 用户实测确认
-- 「🔎 请补充信息」徽章在刷新后丢失（文字还在，语义没了）。
--
-- create_all 只建新表，不会为已存在的表加字段，需手动执行本脚本（MySQL 8）。
-- 用法：mysql -u<user> -p <database> < migrate_clarification_flag.sql
--
-- 幂等性：本脚本**不幂等**，重复执行会报 Duplicate column name，
-- 属预期（与 migrate_feedback_unique.sql 保持一致）。
-- ============================================================

ALTER TABLE chat_messages
  ADD COLUMN is_clarification TINYINT(1) NOT NULL DEFAULT 0
  COMMENT '本条是否为证据不足的追问话术（阶段二）';