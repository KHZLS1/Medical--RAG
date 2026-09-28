"""由「项目开发进度表-通用模板.xlsx」生成「项目开发进度表-医疗RAG.xlsx」。

模板是格式唯一来源（表头/边框/下拉/条件格式/列宽），本脚本只填内容：
  开发进度表  A-E 列  62 张任务卡（11 个阶段）
  进度统计    阶段区按阶段名 + 状态自动汇总，另给两套完成率口径
  使用说明    阶段划分 / 状态口径 / 统计口径 / 维护说明 / 最近更新
"""
from copy import copy

import openpyxl
from openpyxl.styles import Alignment, Font, PatternFill, Side

TPL = (
    r'c:\Users\22413\.trae-cn\attachments\6ab48056b8fd6c94df6a374a'
    r'\496d9439-742f-47b1-992e-13fc8c456d53_项目开发进度表-通用模板.xlsx'
)
OUT = r'c:\Users\22413\Desktop\医疗RAG\项目开发进度表-医疗RAG.xlsx'

# 阶段, 卡号, 任务名称, 开发状态, 备注
tasks = [
    ('P0 环境基建', 'T01', '工程初始化与开发约定（conda + uv）', '已测试', 'Python 3.10；依赖清单与 pyproject 约定'),
    ('P0 环境基建', 'T02', '后端 FastAPI 框架与配置', '已测试', '/api/health 探活正常；.env 驱动配置'),
    ('P0 环境基建', 'T03', 'MySQL 建模与初始化（utf8mb4）', '已测试', 'init_db 建表；字符集 utf8mb4_unicode_ci'),
    ('P0 环境基建', 'T04', 'Milvus 向量库部署（docker-compose）', '已测试', '9091/healthz 就绪；含 etcd 与 MinIO'),
    ('P0 环境基建', 'T05', '前端 Vite/React 工程与 /api 代理', '已测试', '移除干扰 SSE 的 configure 回调'),

    ('P1 数据工程', 'T06', '原始数据清洗（按科室拆分 + 去重）', '已测试', '存 backend/data/by_department/'),
    ('P1 数据工程', 'T07', '数据格式桥梁（字段映射）', '已测试', 'instruction→title 等字段映射'),
    ('P1 数据工程', 'T08', '文本切分与预处理', '已测试', '内置切分 + embedding'),
    ('P1 数据工程', 'T09', '向量化入库（bge-large-zh-v1.5）', '已测试', '全量 60.5 万；先 --limit 500 小批验证'),
    ('P1 数据工程', 'T10', 'BM25 索引与启动预热', '已测试', 'lifespan 预热，k=20 与检索参数一致'),

    ('P2 检索链路', 'T11', '混合检索与动态权重融合', '已测试', '权重优先级 tuned > .env > 默认'),
    ('P2 检索链路', 'T12', '查询改写（多路 + 历史 + 兜底）', '已测试', '修复 quries→queries 拼写'),
    ('P2 检索链路', 'T13', '改写契约封闭化', '已测试', '防说明文字被当检索词'),
    ('P2 检索链路', 'T14', 'Reranker 精排（top_k 配置化）', '已测试', 'FlagEmbedding 依赖'),
    ('P2 检索链路', 'T15', '来源路由 Agent', '已测试', 'LLM 判来源 + 过滤回退'),
    ('P2 检索链路', 'T16', '文档上传与全局去重', '已测试', 'content_hash 唯一索引；多用户化待 T57'),

    ('P3 生成对话', 'T17', 'LLM 接入与降级', '已测试', '实际走硅基流动 OpenAI 兼容网关'),
    ('P3 生成对话', 'T18', '生成链与医学提示词', '已测试', '资料症状不得误归为用户症状'),
    ('P3 生成对话', 'T19', 'SSE 流式回答', '已测试', 'fetch no-store 防 404 缓存'),
    ('P3 生成对话', 'T20', '会话管理与多轮上下文', '已测试', '流开始前 commit 防会话丢失'),
    ('P3 生成对话', 'T21', '前端多轮会话修复', '已测试', '第二问携带同一 conversation_id'),
    ('P3 生成对话', 'T22', '证据二档闸门与兜底节点', '已测试', 'none/strong 二档 + 独立兜底节点'),

    ('P4 对话行为与评测', 'T23', '显式对话行为 dialogue_act', '已测试', '四类行为，零额外延迟'),
    ('P4 对话行为与评测', 'T24', '策略矩阵 dialogue_policy', '已测试', '唯一决策表，替代散落 if'),
    ('P4 对话行为与评测', 'T25', '对话图改造（三出口/两出口）', '已测试', 'ack/chitchat 检索前掉头，省 3~5s'),
    ('P4 对话行为与评测', 'T26', '对话行为评测集与脚本', '已测试', '65 条 9 组对抗集，含 dry-run'),
    ('P4 对话行为与评测', 'T27', '规则门误伤修复', '已测试', '「好恶心/好困」误伤 4 → 0'),
    ('P4 对话行为与评测', 'T28', '检索回归与基线订正', '已测试', '冻结无改写基线 0.92 / 0.9067'),

    ('P5 人工澄清与交互', 'T29', '图状态持久化 checkpointer', '已测试', '同步 Saver 与异步链路不兼容，改 Async'),
    ('P5 人工澄清与交互', 'T30', '人工澄清中断 human_review', '已测试', '10 项验证清单全通过'),
    ('P5 人工澄清与交互', 'T31', '澄清后重检索 requery', '已测试', '不连回子图节点，trace 不翻倍'),
    ('P5 人工澄清与交互', 'T32', '澄清话术落库与刷新恢复', '已测试', 'is_clarification 字段 + 迁移 + 徽章'),
    ('P5 人工澄清与交互', 'T33', '前端复制按钮（5 处）', '已测试', '流式生成中不显示'),
    ('P5 人工澄清与交互', 'T34', 'trace 跨轮累积修复', '已测试', '连续 3 轮每轮 trace 均 5 步'),
    ('P5 人工澄清与交互', 'T35', '验证体系补强（澄清 + 接口）', '已测试', '20/20 + e2e 10 场景 + 33 断言'),

    ('P6 指代消解与忠实性', 'T36', '阶段三·focus_entity 指代消解', '已测试', '仅 followup 且改写退化时兜底'),
    ('P6 指代消解与忠实性', 'T37', '阶段四·忠实性校验', '已测试', 'L2 默认关、fail-open；不进流式节点'),
    ('P6 指代消解与忠实性', 'T38', '阶段三/四单测与回归', '已测试', '全绿退出码 0；端到端验收见 T49'),

    ('P7 前端循证工作台', 'T39', '三栏循证工作台改版', '已测试', '构建通过 + 7 项 UI 验证 PASS'),
    ('P7 前端循证工作台', 'T40', '角标与证据卡双向联动', '已测试', '明暗双主题；emoji 换 SVG 图标'),
    ('P7 前端循证工作台', 'T41', 'Intro 海报页', '已测试', 'localStorage 控制；8 项实测全过'),
    ('P7 前端循证工作台', 'T42', '数据大屏 /#/screen', '已测试', 'stats API + 兜底快照；每屏 14s'),
    ('P7 前端循证工作台', 'T43', '刷新恢复修复（会话 + 状态）', '已测试', '刷新后会话与澄清徽章均保留'),
    ('P7 前端循证工作台', 'T44', '布局缺陷修复（整页撑高）', '已测试', 'grid-template-rows + min-height:0'),
    ('P7 前端循证工作台', 'T45', '消息时间（提问/回复 + 耗时）', '已测试', '刷新前后逐字一致；tsc 0 error'),

    ('P8 全量回归与验证', 'T46', '检索回归复跑', '已测试', '26m24s；0.92 / 0.9067 与基线逐位一致'),
    ('P8 全量回归与验证', 'T47', '对话行为全量回归', '已测试', '1115.3s；act 100% / 落点 95.2%'),
    ('P8 全量回归与验证', 'T48', '改写缓存覆盖核验', '已测试', '54 hits / 0 misses；未命中 11 条为规则门'),
    ('P8 全量回归与验证', 'T49', '阶段三/四端到端验收', '已测试', '09-28 CDP 浏览器验收：阶段三 §3.5 四项 + 阶段四 §3.4 五项全过（含刷新后仍是修正版=命门）+ 阶段二 #3/#9；异常 0/console.error 0'),

    ('P9 交付与发布', 'T50', '引用编号可点击溯源', '已测试', '仅对真实存在的编号生效'),
    ('P9 交付与发布', 'T51', '反馈闭环与 B3 看板', '已开发', '反馈迁移语义已并入 alembic baseline 并 stamp；待人工验收看板'),
    ('P9 交付与发布', 'T52', '仓库整理与发布', '已测试', '修复 clone 跑不起来；master 已同步'),
    ('P9 交付与发布', 'T53', 'README 重写与 .env 纠错', '已测试', '纠正键名为 LLM_API_KEY'),
    ('P9 交付与发布', 'T54', 'AI 工作记忆治理', '已测试', '压缩至 17111 字节；路径改 %USERPROFILE%'),
    ('P9 交付与发布', 'T55', '全栈容器化', '未开发', '目前仅 Milvus 容器化'),

    ('P10 工程化与路线图', 'T56', '改写按对话行为分流', '已测试', 'act 分流默认开启（DIALOGUE_ACT_ENABLED / REWRITE_GATE_ON_ACT）；followup 用改写、其余用原句打分'),
    ('P10 工程化与路线图', 'T57', '多用户鉴权与数据隔离', '未开发', '反馈 stats 目前会泄露他人提问'),
    ('P10 工程化与路线图', 'T58', '数据库迁移工程化（Alembic）', '已测试', 'baseline e03c5380d2c8 + repair 0f6e3b0d0396；真库已收敛，alembic check 零漂移'),
    ('P10 工程化与路线图', 'T59', '生成侧自省：越界编号剥除', '已测试', '原「partial 三档」经实测被排除（top_score 分不开可答/不可答）；改走确定性自省后处理'),
    ('P10 工程化与路线图', 'T60', 'badcase 回流管道与 CI', '已测试', 'export_badcases.py + GitHub Actions 双 job（离线 5 项测试 / ruff E9,F 门）'),
    ('P10 工程化与路线图', 'T61', '长对话历史摘要层', '已测试', '超 6 轮折叠「更早对话要点」；HISTORY_SUMMARY_ENABLED 可回滚'),
    ('P10 工程化与路线图', 'T62', 'LangGraph 深化路线', '已开发', '⑧节点级重试/降级 + ①子图 schema 契约 + ④证据分级回边；6 项离线自测进 CI（④ 默认关，待真实数据标定）'),
]

PHASE_FILL = {
    'P0 环境基建': 'EDE9FE',
    'P1 数据工程': 'DBEAFE',
    'P2 检索链路': 'FEF3C7',
    'P3 生成对话': 'D1FAE5',
    'P4 对话行为与评测': 'FCE7F3',
    'P5 人工澄清与交互': 'CFFAFE',
    'P6 指代消解与忠实性': 'FEE2E2',
    'P7 前端循证工作台': 'E5E7EB',
    'P8 全量回归与验证': 'D9F99D',
    'P9 交付与发布': 'E0E7FF',
    'P10 工程化与路线图': 'FFEDD5',
}
PHASE_RANGE = {
    'P0 环境基建': 'T01–T05', 'P1 数据工程': 'T06–T10', 'P2 检索链路': 'T11–T16',
    'P3 生成对话': 'T17–T22', 'P4 对话行为与评测': 'T23–T28', 'P5 人工澄清与交互': 'T29–T35',
    'P6 指代消解与忠实性': 'T36–T38', 'P7 前端循证工作台': 'T39–T45',
    'P8 全量回归与验证': 'T46–T49', 'P9 交付与发布': 'T50–T55', 'P10 工程化与路线图': 'T56–T62',
}
PHASES = list(PHASE_FILL)

guide = [
    (1,  '医疗RAG 智能问答系统 · 开发进度表', True),
    (3,  '本表按「任务卡驱动」维护：一张任务卡 = 一行进度。内容依据项目历史开发记录整理，截至 2026-09-27。', False),
    (5,  '【阶段划分】', True),
    (6,  'P0 环境基建 —— 工程脚手架、FastAPI 后端框架、MySQL / Milvus、Vite 前端工程', False),
    (7,  'P1 数据工程 —— 数据清洗、格式桥梁、文本切分、向量入库（60.5 万）、BM25 预热', False),
    (8,  'P2 检索链路 —— 混合检索与动态权重、查询改写与契约封闭、Reranker 精排、来源路由 Agent、上传去重', False),
    (9,  'P3 生成对话 —— LLM 接入与降级、医学提示词、SSE 流式、会话多轮、证据二档闸门与独立兜底', False),
    (10, 'P4 对话行为与评测 —— dialogue_act 四类分流、策略矩阵、对话图改造、65 条对抗集、规则门修复、基线订正', False),
    (11, 'P5 人工澄清与交互 —— checkpointer 持久化、human_review 追问、requery 重检索、澄清落库、复制按钮、验证体系补强', False),
    (12, 'P6 指代消解与忠实性 —— 阶段三 focus_entity 跨轮指代消解、阶段四 忠实性校验与越界编号剥除', False),
    (13, 'P7 前端循证工作台 —— 三栏布局、角标与证据卡双向联动、双主题、海报页、数据大屏、布局修复、消息时间', False),
    (14, 'P8 全量回归与验证 —— 检索回归复跑、对话行为全量回归、改写缓存覆盖核验、阶段三/四端到端验收', False),
    (15, 'P9 交付与发布 —— 引用溯源、反馈闭环与看板、仓库发布、README 与 .env 纠错、工作记忆治理、容器化', False),
    (16, 'P10 工程化与路线图 —— 改写按行为分流、多用户鉴权、Alembic 迁移、生成侧自省、badcase 回流、历史摘要、LangGraph 深化', False),
    (18, '【状态流转口径】', True),
    (19, '未开发 → 开发中（任务卡已发出）→ 已开发（代码完成、本卡验收测试全绿，人工验收未做）→ 已测试（人工验收逐条通过、已提交）', False),
    (21, '【进度统计口径】', True),
    (22, '· 全量口径 = 已测试卡数 ÷ 全部卡数（P0~P10）。P10 是尚未启动的工程化待办与路线图，纳入统计会稀释完成率。', False),
    (23, '· 交付口径 = 已测试卡数 ÷ P0~P9 卡数，用于评估「已交付功能」的真实完成度（见「进度统计」页底部）。', False),
    (24, '· 两套口径均由公式驱动，随「开发进度表」状态列实时更新，无需手改。', False),
    (26, '【维护说明】', True),
    (27, '· 「开发状态」列自带下拉与四色条件格式（已预置到第 500 行），新增任务行无需任何设置。', False),
    (28, '· 「进度统计」页全部由公式驱动，按阶段名与状态自动汇总，无需手改任何数字。', False),
    (29, '· 阶段列底色为静态配色（11 个阶段 11 色），新增阶段可手动填充颜色区分。', False),
    (30, '· 新增行后如筛选范围不对，请重新点一次「数据 → 筛选」。', False),
    (32, '【最近更新 · 2026-09-27 夜】', True),
    (33, '· 完成「离线可完成批」6 项工程化：T59 生成侧自省剥编号、改写缓存冻结加载、checkpoint 快照清理、Alembic 迁移工程化、badcase 回流 + CI、长对话历史摘要。', False),
    (34, '· 5 项离线测试全绿（退出码 0）；GitHub Actions 双 job（离线测试 + ruff E9,F）落地。', False),
    (35, '· Alembic 两条 revision 落地（baseline e03c5380d2c8 + repair 0f6e3b0d0396）；真库 21 条孤儿消息已清并补上外键，alembic check 零漂移。', False),
    (36, '· T59 经实测否定原「partial 三档」方案（top_score 分不开可答/不可答），改为确定性自省后处理，规则 9/10 一个字未动。', False),
    (37, '· 仍待办：T57 多用户鉴权、T49 全栈端到端验收、T55 全栈容器化、T62 LangGraph 路线深化。', False),
    (39, '【最近更新 · 2026-09-28】', True),
    (40, '· T49 完成：本机 Chrome + CDP 无头驱动做浏览器端到端验收，阶段三四项、阶段四五项、阶段二 #3/#9 全过。', False),
    (41, '· 命门项实测通过：注入越界 [9] → 页面与复制文本均无 [9]，且**刷新后原文逐字一致**（证明 correction 覆盖落库生效）。', False),
    (42, '· 对抗集扩到 71 条 / 10 组（新增 focus_entity 组 6 条），冻结改写缓存同步扩到 66 条。', False),
    (43, '· 暴露一处基础设施问题（非应用 bug）：provider 尾延迟 1.4s/33.6s/254.1s，前端 120s 整轮硬超时会误杀仍在生成的流。', False),
    (44, '· 仍待办：T57 多用户鉴权、T55 全栈容器化、T62 LangGraph 路线深化。', False),
    (46, '【最近更新 · 2026-09-28 午】T62 LangGraph 深化（三件一起做）', True),
    (47, '· ⑧ 节点级可靠性：按类别挂 RetryPolicy、按节点名挂 error_handler。两个实测硬结论——流式节点不能挂重试（会把失败那次的残 token 和重试结果拼一起）、error_handler 是 PUSH 任务不走出边（要续跑必须 Command(goto=)）。', False),
    (48, '· ① 子图 schema 契约：RetrievalInput 刻意不含 trace ⇒ 解除「父图在子图前不许写 trace」的约束；连带删掉 requery 节点（程序化调用 → 普通边）。', False),
    (49, '· ④ 证据分级回边（Self-RAG 式，默认关）：grade 判「资料够不够回答」，不够就换检索词回边重检；重试轮改写直通，不写第二条「意图判定」。', False),
    (50, '· 新增 6 项离线自测（python scripts/test_node_reliability.py）并进 CI；既有 5 项离线测试全绿，ruff E9,F 零告警。', False),
    (51, '· 仍待办：T57 多用户鉴权、T55 全栈容器化；④ 需在真实数据上标定后再决定是否默认打开。', False),
    (52, '【最近更新 · 2026-09-28 下午】T62 实测暴露并修掉两个缺陷', True),
    (53, '· 🔴 节点级 TimeoutPolicy 会让后端起不来：langgraph 在 compile() 阶段直接抛 ValueError（超时只支持 async 节点），本项目全是 sync，而默认值 0 恰好掩盖了它。已删除该机制，换成真正生效的客户端级 LLM_TIMEOUT_SEC（透传 ChatOpenAI(timeout=)），并加防地雷回归断言。', False),
    (54, '· 🔴 ④ 的回边拿不到新词：靶样本 4/4 的 retry_query 与当前检索词逐字相同（分级 Prompt 看不到上一次用的词 ⇒ 必然与改写节点撞词），检索确定性 ⇒ 同词重跑结果不变，那一轮是纯浪费。经拍板**直接砍掉整条回边**（连守卫一齐删，见下条晚段），只保留「判不足 → 走无资料路径」这半边。', False),
    (55, '· ④ 标定实得：开启后 ooc-04/05/08/09（top_score 0.576/0.330/0.914/0.647）全部从 generate 翻成 insufficient ⇒ 确实做到了分数做不到的事；净收益全部来自「判不足 → 走无资料路径」这半边。', False),
    (56, '· 全量复跑 71 条（④ 关）与基线逐位一致：act 59/59、落点 60/63、方向性错误 0、缓存 66/59/0 ⇒ 删 requery + 加 grade 对默认路径零副作用。', False),
    (57, '· 离线自测 6 项全绿（node_reliability 116 项断言），ruff E9,F 零告警。', False),
    (58, '· 仍待办：④ 的回边要真正有效需改分级 Prompt（把 enhanced_query 喂进去并要求避开已用词）；sm-08/gm-03 因 provider 劣化未跑完；流式调用没有硬墙钟（client timeout 会被持续吐字重置）。', False),
    (59, '【最近更新 · 2026-09-28 晚】T62-B 收口：砍回边 + 流式空闲看门狗', True),
    (60, '· B-1 砍掉 ④ 回边：_same_query 守卫**本身是死代码**（键在 verdict=="retry" 上，而 parse_grade_output 只认字面 insufficient，其余一律折成 sufficient ⇒ 永不触发），已随整条回边一齐删除。node_grade 现在只做一件事：判不足就把这轮推进无资料路径；GraphState/RetrievalInput 的 grade_retry 字段、_route_after_grade、node_rewrite 的「重检索轮直通」分支全部清掉。', False),
    (61, '· B-3 流式空闲看门狗（后端）：main.py 新增可离线单测的 _watchdog_iter，按**空闲**超时（SSE_IDLE_TIMEOUT_SEC，默认 90s）逐条取流；超时则补兜底文案（区分「一个字都没到」与「吐了一半」两段，都指向急诊）并推 correction 事件，复用既有「整段替换 + 落库覆盖」通道（verdict=stream_idle_timeout）。', False),
    (62, '· B-3 前端：Chat.tsx 从「整轮 120s 硬超时」改为「空闲 105s」（每收到一条事件重新计时，patchLast/onToken 各 touchIdle 一次）。阈值刻意大于后端 90s，让后端先收尾。这才是真正的修法——sse_starlette 每 15s 发 : ping，连接全程是活的，整轮计时会误杀仍在生成的流。', False),
    (63, '· 新增第 7 项离线自测 test_sse_idle.py（7 组断言；关键护栏 B3-3：总耗时 > idle 但每条间隔 < idle 的流一条都不丢，防「整轮计时」退化），已进 CI；test_node_reliability.py 在 B-1 后 116 项 PASS / 退出码 0，既有 5 项离线测试全绿，ruff E9,F 零告警。', False),
    (64, '· 收口验证（09-28 晚全量复跑 71 条，冻结缓存）：act 59/59 = 100%、落点 60/63 = 95.2%、方向性错误 0、规则门 12/71 零误伤、缓存 66/59/0 frozen、622.8s（8.8s/条）⇒ 与改动前那轮**逐条字段级 0 差异**（act/route/gated/evidence/top_score/enhanced_query/scored_by/steps 全同），证明 B-1/B-3 对默认路径**零副作用**。', False),
    (65, '· sm-08「好痛」0.501 / gm-03「好难受」0.523 本轮补跑成功（此前卡在 provider）⇒ 与 ooc-05（0.330）同源：裸症状主诉/库外问题拿了 0.33~0.52 的沾边分被判 strong 后走 generate，属**阈值问题非分类问题**；抬 RERANK_SCORE_THRESHOLD 会误杀 0.658 的真·可回答样本，正解是 ④ 的证据分级。', False),
    (66, '· ⚠️ 仍存：流式调用没有硬墙钟（LLM_TIMEOUT_SEC 会被持续吐字/心跳重置，不是墙钟上限），B-3 的 idle 看门狗是**产品层兜底而非根治**；④ 若将来要复活回边，必须先解决「分级 Prompt 看不到上一次检索词」这条根因。EVIDENCE_GRADE_ENABLED 保持默认关（已拍板）。', False),
]

wb = openpyxl.load_workbook(TPL)

# ---------- 开发进度表 ----------
ws = wb['开发进度表']
base = {c: copy(ws.cell(2, c)._style) for c in range(1, 6)}
last = 1 + len(tasks)

for r in range(2, last + 4):
    for c in range(1, 6):
        ws.cell(r, c)._style = copy(base[c])
    ws.row_dimensions[r].height = 26

for i, (phase, card, name, status, note) in enumerate(tasks):
    r = 2 + i
    ws.cell(r, 1).value = phase
    ws.cell(r, 2).value = card
    ws.cell(r, 3).value = name
    ws.cell(r, 4).value = status
    ws.cell(r, 5).value = note
    ws.cell(r, 1).fill = PatternFill('solid', fgColor=PHASE_FILL[phase])

for r in range(last + 1, last + 4):
    for c in range(1, 6):
        ws.cell(r, c).value = None
    ws.cell(r, 1).fill = PatternFill('solid', fgColor='FFFFFF')

ws.column_dimensions['A'].width = 22   # 「P10 工程化与路线图」需 20 个单位
ws.column_dimensions['C'].width = 60
ws.column_dimensions['E'].width = 52

# ---------- 进度统计 ----------
st = wb['进度统计']
src = {c: copy(st.cell(12, c)._style) for c in range(1, 9)}
st.cell(10, 1).value = '阶段进度（按阶段名自动汇总；阶段数不同则插入/删除行并整行复制公式）'

for idx, name in enumerate(PHASES):
    r = 12 + idx
    for c in range(1, 9):
        st.cell(r, c)._style = copy(src[c])
    st.cell(r, 1).value = name
    st.cell(r, 1).fill = PatternFill('solid', fgColor=PHASE_FILL[name])
    st.cell(r, 2).value = PHASE_RANGE[name]
    st.cell(r, 3).value = f'=COUNTIF(开发进度表!A:A,A{r})'
    st.cell(r, 4).value = f'=COUNTIFS(开发进度表!A:A,A{r},开发进度表!D:D,"已测试")'
    st.cell(r, 5).value = f'=COUNTIFS(开发进度表!A:A,A{r},开发进度表!D:D,"已开发")'
    st.cell(r, 6).value = f'=COUNTIFS(开发进度表!A:A,A{r},开发进度表!D:D,"开发中")'
    st.cell(r, 7).value = f'=COUNTIFS(开发进度表!A:A,A{r},开发进度表!D:D,"未开发")'
    st.cell(r, 8).value = f'=TEXT(D{r}/MAX(1,C{r}),"0.0%")'

tr = 12 + len(PHASES)          # 合计行
for c in range(1, 9):
    cell = st.cell(tr, c)
    cell._style = copy(src[c])
    cell.font = Font(name='微软雅黑', size=10, bold=True)
    border = copy(cell.border)
    border.top = Side(style='thin', color='9CA3AF')
    cell.border = border
st.cell(tr, 1).value = '合计'
st.cell(tr, 1).fill = PatternFill('solid', fgColor='F9FAFB')
st.cell(tr, 2).value = f'T01–T{len(tasks):02d}'
for col, letter in ((3, 'C'), (4, 'D'), (5, 'E'), (6, 'F'), (7, 'G')):
    st.cell(tr, col).value = f'=SUM({letter}12:{letter}{tr - 1})'
st.cell(tr, 8).value = f'=TEXT(D{tr}/MAX(1,C{tr}),"0.0%")'

# 两套完成率口径
st.cell(25, 1)._style = copy(st.cell(10, 1)._style)
st.cell(25, 1).value = '完成率口径（P10 为路线图待办，纳入统计会稀释完成率）'
st.cell(25, 1).font = Font(name='微软雅黑', size=11, bold=True)

for r, label, formula, note, fill in (
    (26, '口径A', '=TEXT(COUNTIF(开发进度表!D:D,"已测试")/MAX(1,COUNTA(开发进度表!B:B)-1),"0.0%")',
     '全量 P0~P10', 'F3F4F6'),
    (27, '口径B',
     '=TEXT((COUNTIF(开发进度表!D:D,"已测试")-COUNTIFS(开发进度表!A:A,"P10 工程化与路线图",开发进度表!D:D,"已测试"))'
     '/MAX(1,COUNTA(开发进度表!B:B)-1-COUNTIF(开发进度表!A:A,"P10 工程化与路线图")),"0.0%")',
     '交付 P0~P9', 'D1FAE5'),
):
    for c in range(1, 9):
        st.cell(r, c)._style = copy(src[c])
    st.cell(r, 1).value = label
    st.cell(r, 1).font = Font(name='微软雅黑', size=10, bold=True)
    st.cell(r, 1).fill = PatternFill('solid', fgColor='FFFFFF')
    st.cell(r, 2).value = formula
    st.cell(r, 2).font = Font(name='微软雅黑', size=10, bold=True)
    st.cell(r, 2).fill = PatternFill('solid', fgColor=fill)
    st.cell(r, 3).value = note

st.column_dimensions['A'].width = 22
st.column_dimensions['B'].width = 14

# ---------- 使用说明 ----------
gs = wb['使用说明']
base_guide = copy(gs.cell(18, 1)._style)
guide_rows = {g[0] for g in guide}
for r, text, bold in guide:
    cell = gs.cell(r, 1)
    cell._style = copy(base_guide)
    cell.value = text
    cell.font = Font(name='微软雅黑', size=12 if (bold or r == 1) else 10, bold=bold)
    cell.alignment = Alignment(horizontal='left', vertical='center', wrap_text=True)
for r in range(1, 68):
    if r not in guide_rows:
        gs.cell(r, 1).value = None

wb.calculation.fullCalcOnLoad = True
wb.save(OUT)

# ---------- 自检（独立于 Excel 公式重算） ----------
from collections import Counter
by_phase = Counter(t[0] for t in tasks)
tested = Counter(t[0] for t in tasks if t[3] == '已测试')
total = len(tasks)
p10 = by_phase['P10 工程化与路线图']
tested_all = sum(tested.values())
tested_p09 = tested_all - tested['P10 工程化与路线图']
print('saved:', OUT)
print('任务卡:', total, '| 阶段:', len(PHASES), '| 末行:', last)
print('已测试:', tested_all, '| 已开发:', sum(1 for t in tasks if t[3] == '已开发'),
      '| 开发中:', sum(1 for t in tasks if t[3] == '开发中'),
      '| 未开发:', sum(1 for t in tasks if t[3] == '未开发'))
print(f'口径A 全量   : {tested_all}/{total} = {tested_all / total * 100:.1f}%')
print(f'口径B 交付   : {tested_p09}/{total - p10} = {tested_p09 / (total - p10) * 100:.1f}%')
for name in PHASES:
    print(f'  {name}: {by_phase[name]} 卡 / 已测试 {tested[name]}')