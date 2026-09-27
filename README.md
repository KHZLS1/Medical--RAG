# 医疗智能问答系统（Medical RAG）

基于 **LangGraph + FastAPI + Milvus + 硅基流动 LLM** 的多轮医疗问答系统。
知识库来自开源中文医疗问答数据集（清洗后约 60.5 万条）。

不只做「检索 → 生成」那条直线：多轮对话里真正难的是**这句话该不该检索、用哪句去检索、
检索不到时该说什么**。本项目把这些判断收敛成一张策略表 + 一张 LangGraph 状态图，
并在生成侧加了确定性的引用编号校验。

> ⚠️ 接口**没有鉴权**，仅适合本机 / 内网 demo，不要直接暴露到公网。

---

## 目录

- [技术栈](#技术栈)
- [系统架构](#系统架构)
- [对话行为策略矩阵](#对话行为策略矩阵)
- [数据与 Schema](#数据与-schema)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [配置项速查](#配置项速查)
- [接口一览](#接口一览)
- [评估与调优](#评估与调优)
- [测试与自测](#测试与自测)
- [运维与排错](#运维与排错)
- [医疗安全提示](#医疗安全提示)
- [已知限制](#已知限制)

---

## 技术栈

| 层 | 选型 |
|---|---|
| 前端 | React 18 + Vite + TypeScript（SSE 打字机、引用溯源、展示大屏） |
| 编排 | **LangGraph 状态图**（条件路由 + checkpointer + `interrupt()` 人工澄清） |
| 后端 | FastAPI + SQLAlchemy + SSE（`sse-starlette`） |
| 向量库 | Milvus 2.5+（服务端 BM25 Function + 稠密向量 `hybrid_search`） |
| Embedding | `BAAI/bge-large-zh-v1.5`（1024 维） |
| Reranker | `BAAI/bge-reranker-v2-m3`（Cross-Encoder 精排，`FlagEmbedding`） |
| 生成 | 硅基流动（SiliconFlow）OpenAI 兼容网关，默认 `deepseek-ai/DeepSeek-V4-Flash` |
| 会话存储 | MySQL 8（会话 / 消息 / 上传文档元数据 / 反馈） |
| 检查点 | SQLite（`langgraph-checkpoint-sqlite`，供澄清中断恢复） |

---

## 系统架构

### LangGraph 状态图

```
                                   ┌── chat ──────────────────────────────→ node_chat_generate ─→ END
intent（纯规则 + LLM 分类） ───────┤                                        （不检索、不标引用）
                                   └── medical → 检索子图
                                                  rewrite（改写 + 顺带产出 dialogue_act / focus_entity）
                                                    ├─ act = ack / chitchat →（子图结束）→ chat
                                                    └─ 其余 → route → retrieve → rerank
                                                                                    ├─ strong → generate → ground_check → END
                                                                                    └─ none ─┬─ 可追问且非急症 → human_review（interrupt 暂停）
                                                                                             │        └─ 用户补充 → requery ─→ …
                                                                                             └─ 否则 → insufficient → END
```

- **单一事实来源**：检索子图是唯一的检索实现，`rag_chain.retrieve()` 直接调用它，
  不存在「图」与「旧函数」两份逻辑漂移。意图分流**不在**子图内 —— `eval_rag.py`
  直接调子图做检索评估，不希望被「这题算不算医学问题」干扰。
- **证据只有两档**：`strong` / `none`。实测 rerank 分数双峰分布，中间是空档，
  按区间切 `partial` 永远不触发（字段留位待标定）。
- **无证据路径整条挪出 `MEDICAL_PROMPT`**：该 Prompt 的规则「必须标注引用编号 + 免责声明」
  是无条件的，context 为空时模型只能硬编一个 `[1]`。因此改走独立的
  `INSUFFICIENT_PROMPT`（禁引用 + **代码确定性追加**免责声明），而不是去改那两条规则。
- **检查点与人工澄清**：`evidence=none` 且非急症且轮次未用完时，`node_human_review`
  调 `interrupt()` 暂停整张图（状态落盘 SQLite），SSE 以 `clarification_request` 收尾；
  下一轮请求由 `astream_resume` 以 `Command(resume=...)` 恢复。单轮最多追问
  `HUMAN_REVIEW_MAX_ROUNDS` 次，防「追问 → 还是没有 → 再追问」死循环。
- **忠实性校验**：`generate → ground_check`。L1 用正则校验回答里的 `[n]` 是否都能在
  sources 里找到，越界编号由程序**确定性剥除**（不靠 Prompt 求模型守规矩）；
  再加一层**生成侧自省**：回答开头就自认「资料未提供 / 无法回答」时，编号是真的、
  但整篇没有可引之处，于是**整篇剥掉**（`unanswerable_stripped`）。
  L2（逐句核查论断是否有 context 支持，默认关）开启时额外调一次 LLM。
  文本真的变了才推 `correction` 事件，前端整段替换。

> 设计决策与踩坑记录见根目录 `实施计划_阶段二_检查点与人工澄清.md`、
> `实施计划_阶段三_指代消解.md`、`实施计划_阶段四_忠实性校验.md`
> （⚠️ 这些设计文档已被 `.gitignore` 排除，只存在于本机工作副本，不随仓库分发）。

### 检索链路

```
用户提问
  └─ 查询改写（LLM，温度 0.0）—— 同时产出 dialogue_act 与 focus_entity
       └─ 混合检索（pymilvus hybrid_search，各召回 20 条）
            ├─ 稀疏：query 原文 → Milvus 服务端 BM25（sparse 字段，由 Function 生成）
            ├─ 稠密：bge 编码 → embedding 字段（metric=IP）
            └─ 融合：WeightedRanker(BM25, 向量)，norm_score=True，权重可配（也可切 RRFRanker）
       └─ Reranker 精排（bge-reranker-v2-m3，Top-K 默认 5，normalize=True）
            └─ 流式生成（严格基于检索资料，附引用编号与免责声明）
```

**症状归属约束**：生成 Prompt 注入【用户情况】（仅取历史中用户亲自描述的主诉），
强制区分「用户症状」与「资料中该疾病的症状表述」，防止把资料里的症状误当作用户主诉。

### 模型离线加载（重要）

断网/国内直连环境下的关键约束：

1. `app/__init__.py` 在任何 `transformers` / `huggingface_hub` 子模块被导入前设置
   `HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1`（这两个是**模块级常量**，import 时定型，
   放到 `config.py` 里再设已经晚了）。
2. `config.resolve_hf_snapshot()` 把模型 id 解析成**本地快照绝对路径**
   （`refs/main` → `snapshots/` 最新目录 → 回退原 id），不硬编码 hash。
3. `reranker.py` 交绝对路径后，`from_pretrained` 直接走 `os.path.isdir` 分支，零网络请求。

> 为什么必须这么做：`FlagReranker` 虽然声明了 `**kwargs`，但构造 tokenizer/model 时
> **只下传** `trust_remote_code` / `cache_dir`，其余 kwargs 全部丢弃 —— 于是进程照样向
> huggingface.co 发 HEAD 校验版本，国内直连在 TLS 阶段被阻断（`SSLEOFError`）。
> 另注意参数名是 **`devices`（复数）**，写成 `device=` 会被 kwargs 吞掉、静默失效。

---

## 对话行为策略矩阵

`dialogue_act` 由**改写那一次 LLM 调用顺带产出**（零额外延迟），判据统一收敛在
`app/dialogue_policy.py`，`graph.py` 里不散落字符串比较。

| dialogue_act | 检索 | 检索用 | rerank 打分用 | 证据 strong | 证据 none | 焦点实体 |
|---|---|---|---|---|---|---|
| `new_question` | 是 | 原 question | 原 question | MEDICAL（引用+声明） | INSUFFICIENT（无引用+声明） | update |
| `followup` | 是 | enhanced_query | enhanced_query | MEDICAL（引用+声明） | INSUFFICIENT（无引用+声明） | update |
| `ack` | 否 | — | — | CHAT（无引用无声明） | 同左 | keep |
| `chitchat` | 否 | — | — | CHAT（无引用无声明） | 同左 | keep |

三条必须**同时**成立的不等式，否则已修的 bug 会回归：

1. `followup` 用改写 query 检索 + 打分 —— 指代被补全后才有可比性
   （「它有什么副作用」按原句打分只有 0.355，会被阈值挡成「没资料」）。
2. `ack` / `chitchat` 在**检索前**掉头 —— 否则「不了」被改写补全成病症查询后，
   第 1 条的「改用改写打分」会让它**高分命中**，比原 bug 更隐蔽。
3. `new_question` 用**原句**检索（丢弃改写产物）—— 自足问题上改写是**净负收益**：
   实测无改写 `hit_rate` 0.92 / `mrr` 0.9067，有改写 0.70~0.82（12 题有差异、
   0 题反向，符号检验 p≈0.016）。

第 3 条与第 1 条方向相反，是刻意的不对称：**改写只服务于「把省略/指代补全」，
不服务于「把问题重述一遍」**。

未知 / 解析失败的 act 一律回退 `new_question`（漏答比多答贵）。但
`should_update_focus()` 刻意用**精确成员判定** —— 回退语义在「要不要检索」上保守，
在「要不要覆盖跨轮状态」上恰恰是污染源。

**回滚开关**：`DIALOGUE_ACT_ENABLED=false` → act 恒 `new_question`（完全复原还需一并
回滚改写 Prompt）；`FOLLOWUP_SCORE_WITH_ENHANCED` 单独控制 followup 是否用改写 query 打分；
`REWRITE_GATE_ON_ACT=false` 退回「改写一律生效」。

---

## 数据与 Schema

**Milvus collection**（由 `scripts/ingest.py` 建立，字段名与代码强绑定）

| 字段 | 类型 | 说明 |
|---|---|---|
| `id` | INT64, auto_id | 主键 |
| `embedding` | FLOAT_VECTOR(1024) | bge 稠密向量，AUTOINDEX / IP |
| `content` | VARCHAR(65535) | 原文，`enable_analyzer` + chinese 分词 |
| `sparse` | SPARSE_FLOAT_VECTOR | **由 BM25 Function 服务端生成，客户端不写** |
| `metadata` | JSON | `department` / `title` / `source` / `filename` |

**检索参数（已动态化，免改代码）**：BM25/向量融合权重、Reranker Top-K 均可在
`backend/app/config.py` 或 `.env` 配置；`eval_rag.py tune` 搜出的最优权重写入
`backend/data/tuned_weights.json`，后端启动时优先采纳。

权重优先级：`tuned_weights.json` > `.env` > 代码默认值。
召回 `k` 默认 20，融合后给 Reranker 的候选数为 `min(2k, 50)`。
> 改动权重后需重启后端（`get_retriever` 带 `lru_cache`）。

**MySQL 表**：`conversations` / `chat_messages` / `uploaded_documents` / `feedback`。
表结构声明在 `app/models.py`，变更由 **alembic** 管（见「运维与排错」）。
后端启动仍会 `create_all` 兜底建表（幂等），但**改结构请走迁移**：
只改 models 不会更新已存在的表 —— 项目里"真库比 models 多出两个索引、少一个外键"
那类漂移就是这么来的（已于 `0f6e3b0d0396` 收敛，现 `alembic check` 零漂移）。
**诊断漂移用 `python -m alembic check`**，比肉眼比对 information_schema 靠谱。

---

## 项目结构

```
医疗RAG/
├── Chinese-medical-dialogue-data-master/   # 原始数据集（GBK，不入库）
├── backend/
│   ├── app/
│   │   ├── __init__.py         # ★ 最早的离线开关：HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE
│   │   ├── main.py             # FastAPI 入口：/chat /ingest /ingest-stream /health
│   │   ├── config.py           # 配置 + resolve_hf_snapshot()
│   │   ├── graph.py            # ★ LangGraph 状态图：全部节点、路由、trace 不变量
│   │   ├── dialogue_policy.py  # ★ 对话行为 × 证据 → 答案策略的唯一决策表
│   │   ├── intent.py           # 规则门：纯规则判定会话语（对表外症状词同样成立）
│   │   ├── query_rewriter.py   # 查询改写 + dialogue_act/focus 解析 + 会话标题
│   │   ├── rewrite_cache.py    # 改写结果落盘复用（评估可复现的唯一来源）
│   │   ├── checkpointer.py     # AsyncSqliteSaver 装配
│   │   ├── vectorstore.py      # ★ 检索与写入核心（pymilvus 原生混合检索）
│   │   ├── reranker.py         # bge-reranker-v2-m3 精排 + 去重 + 科室过滤
│   │   ├── rag_chain.py        # 三条生成链 + Prompt + 引用编号校验
│   │   ├── llm.py              # OpenAI 兼容客户端（timeout=180, max_retries=1）
│   │   ├── agents/source_router.py  # 知识来源路由（数据集 / 上传文档）
│   │   ├── stats.py            # 展示大屏聚合（语料扫描 / 入库量 / 评估指标）
│   │   ├── data_loader.py      # CSV/JSON 数据集加载 + 上传文件读写
│   │   ├── text_split.py       # 中文滑窗切分
│   │   ├── database.py / models.py
│   │   └── api/
│   │       ├── documents.py     # 上传（内容哈希去重）、列表、删除（联动清理向量）
│   │       ├── conversations.py # 会话增删改查
│   │       ├── feedback.py      # 反馈（赞/踩、修正答案、备注、统计）
│   │       └── stats.py         # GET /api/stats/overview（大屏单接口）
│   ├── alembic/                # ★ 数据库迁移（env.py 复用 app 的 settings 与 Base.metadata）
│   │   └── versions/           #   baseline + repair 两条 revision（详见「运维与排错」）
│   ├── alembic.ini             # ⚠️ 保持纯 ASCII：alembic 用本地码页（中文 Windows 是 GBK）读它
│   ├── scripts/                # 见下表
│   ├── requirements.txt
│   └── .env.example
├── frontend/                    # React + Vite + TS
│   └── src/
│       ├── App.tsx              # 视图切换 / 主题 / 会话选中态持久化
│       ├── components/Chat.tsx  # 主对话区（SSE 消费、引用溯源渲染）
│       ├── components/EvidencePanel.tsx  # 检索过程 / 来源卡片
│       ├── components/Screen.tsx         # 展示大屏（无人值守轮询）
│       ├── components/FeedbackPanel.tsx  # 反馈看板
│       ├── api/chat.ts / api/stats.ts
│       └── utils/time.ts        # 时间显示口径（见「运维与排错」）
├── docker-compose.yml           # Milvus standalone + etcd + minio
├── .github/workflows/ci.yml     # CI：5 个离线自测 + ruff(E9,F)
└── 实施计划_阶段{二,三,四}_*.md   # 各阶段设计与决策记录（本地工作副本，已被 .gitignore 排除）
```

**backend/scripts/**

| 脚本 | 用途 |
|---|---|
| `preflight_check.py` | 入库前置体检（只读：环境 / schema / 依赖 / 资源） |
| `ingest.py` | 建 collection + 批量入库（**默认先 drop 再建**，追加用 `--no-recreate`） |
| `clean_data.py` | 原始 CSV → 按科室清洗 JSON |
| `drop_collection.py` | 清空 `medical_qa` 集合 |
| `init_db.py` / `check_mysql.py` | 建库 / 检查 MySQL 连接 |
| `check_data.py` | 查看库内数据、科室分布、检索自测 |
| `eval_rag.py` | 检索质量评估 + 权重网格搜索 + 基线管理 |
| `eval_dialogue.py` | 对话行为分流评测（对抗样本集混淆矩阵 + 落点正确率） |
| `benchmark.py` | 分环节测速（GPU embedding vs Milvus 写入） |
| `gen_dashboard_snapshot.py` | 生成大屏兜底快照 |
| `freeze_rewrite_cache.py` | 把当前改写缓存冻结成**入库快照**（A/B 两边吃同一份改写） |
| `export_badcases.py` | 反馈差评 → JSONL 素材（badcase 回流，导完需人工筛选再进评估集） |
| `cleanup_orphan_messages.py` | 审计指向已删会话的孤儿消息（默认 dry-run + 自动备份）；应急清理加 `--apply` |
| `test_*.py` | 离线自测：澄清流程 / 指代消解 / 忠实性校验 / 改写缓存 / 历史摘要 / lifespan / SSE 代理 / 接口回归（哪些进 CI 见「测试与自测」） |
| `migrate_feedback.py` / `migrate_clarification_flag.sql` | **【已废弃】** 手写 SQL 迁移，语义已并入 alembic baseline，留作历史记录 |
| `gen_bm25_cache.py` | **【已废弃】** 旧的客户端 BM25 缓存脚本 |

---

## 快速开始

### 0) 环境要求

- Docker（跑 Milvus）、MySQL 8（跑会话存储）
- Python 3.10+
- 模型：`bge-large-zh-v1.5`（约 1.3GB）+ `bge-reranker-v2-m3`（约 2GB）
  - **首次运行需联网下载**并写入 HuggingFace 本地缓存（`~/.cache/huggingface/hub/`）。
  - 国内直连被阻断时，先设镜像再跑一次脚本即可落缓存：
    `export HF_ENDPOINT=https://hf-mirror.com`。
  - 缓存就绪后代码**强制离线加载**（见上文「模型离线加载」），不再访问 huggingface.co。

### 1) 启动 Milvus（需 2.5+，内置 BM25 Function）

```bash
docker compose up -d
# 等待 30s+，访问 http://localhost:9091 查看健康状态
```

### 2) 配置后端

```bash
cd backend
python -m venv .venv && .\.venv\Scripts\Activate.ps1   # Windows
pip install -r requirements.txt

cp .env.example .env
# 必填：LLM_API_KEY、DATABASE_URL（MySQL 连接串）
python scripts/init_db.py        # 创建 medical_rag 数据库（表在后端启动时自动建）
```

### 3) 数据入库

```bash
python scripts/preflight_check.py                 # 前置体检（只读）
python scripts/ingest.py --cleaned --limit 500    # 冒烟：每科 500 条
python scripts/ingest.py --cleaned                # 全量：约 60.5 万条（CPU 耗时较长）
python scripts/ingest.py --cleaned --dept 内科    # 只入指定科室
python scripts/ingest.py --cleaned --no-recreate  # 追加写入，不清空
```

> 清洗数据在 `backend/data/by_department/*.json`，由 `scripts/clean_data.py` 生成。
> ⚠️ `backend/data/` 被 `.gitignore` 排除，但其中的**评估输入**（冻结测试集、对抗集、
> 基线、`tuned_weights.json`）已用 `git add -f` 强制入库 —— 丢了拿不回来，别清。

### 4) 启动前后端

```bash
# 终端 1：后端
cd backend && uvicorn app.main:app --reload --port 8000

# 终端 2：前端
cd frontend && npm install && npm run dev
```

打开 http://localhost:5173 即可提问（前端已配好 `/api` 代理到 8000）。
**首次提问会加载 Embedding 与 Reranker 模型，等几十秒属正常。**

---

## 配置项速查

全部在 `backend/.env`（`app/config.py` 里有默认值，`extra="ignore"`）。

| 变量 | 默认 | 说明 |
|---|---|---|
| `LLM_API_KEY` | — | **必填**，硅基流动 Key |
| `LLM_BASE_URL` | `https://api.siliconflow.cn/v1` | OpenAI 兼容网关 |
| `LLM_MODEL` | `deepseek-ai/DeepSeek-V4-Flash` | 生成模型 |
| `DATABASE_URL` | — | **必填**，MySQL 连接串 |
| `MILVUS_HOST` / `MILVUS_PORT` / `MILVUS_COLLECTION` | `localhost` / `19530` / `medical_qa` | |
| `EMBEDDING_DEVICE` | `cpu` | 改 `cuda` 可提速 20~40 倍 |
| `BM25_WEIGHT` / `VECTOR_WEIGHT` | `0.3` / `0.7` | 混合检索融合权重 |
| `RERANKER_TOP_K` | `5` | 精排最终保留条数 |
| `QUERY_REWRITE_ENABLED` | `true` | 关掉省一次 LLM 调用，但召回质量下降 |
| `REWRITE_TEMPERATURE` | `0.0` | 改写**必须零温**；0.3 会让同题每次改写都不同 |
| `REWRITE_CACHE_ENABLED` / `REWRITE_CACHE_PATH` | `true` / `data/cache/rewrite_cache.json` | 改写落盘复用 |
| `REWRITE_CACHE_FROZEN` / `REWRITE_FROZEN_PATH` | `false` / `data/frozen/rewrite_frozen.json` | 冻结改写集：跳过指纹校验、只读加载，让 A/B 两边吃同一份改写（生成见 `scripts/freeze_rewrite_cache.py`） |
| `HISTORY_SUMMARY_ENABLED` | `true` | 长对话历史摘要层：超出窗口的旧发言压成一行要点（纯确定性、零 LLM） |
| `RELEVANCE_GATE_ENABLED` / `RERANK_SCORE_THRESHOLD` | `true` / `0.30` | 相关性闸门：低于阈值判「库中无相关内容」 |
| `INTENT_GATE_ENABLED` / `INTENT_CHAT_MAX_CHARS` | `true` / `8` | 规则门：会话语短路 |
| `DIALOGUE_ACT_ENABLED` | `true` | 对话行为分流总开关 |
| `FOLLOWUP_SCORE_WITH_ENHANCED` | `true` | followup 是否用改写 query 打分 |
| `REWRITE_GATE_ON_ACT` | `true` | `new_question` 是否跳过改写用原句检索 |
| `FOCUS_ENTITY_ENABLED` | `true` | 跨轮焦点实体（指代消解），依赖 `DIALOGUE_ACT_ENABLED` |
| `GRAPH_CHECKPOINTER_ENABLED` / `GRAPH_CHECKPOINTER_DB_PATH` | `true` / `data/cache/graph_checkpoints.sqlite` | |
| `HUMAN_REVIEW_ENABLED` / `HUMAN_REVIEW_MAX_ROUNDS` | `true` / `1` | 证据不足时中断追问 |
| `GROUNDEDNESS_CITATION_CHECK` | `true` | L1：越界引用编号确定性剥除 |
| `GROUNDEDNESS_LLM_ENABLED` | `false` | L2：逐句忠实性核查（+1 次 LLM 调用，确认误报率后再开） |
| `UPLOAD_MAX_SIZE_MB` / `UPLOAD_ALLOWED_EXTENSIONS` | `100` / `.pdf,.docx,.txt,.md,.csv` | |
| `BACKEND_HOST` / `BACKEND_PORT` / `FRONTEND_URL` | `0.0.0.0` / `8000` / `http://localhost:5173` | |

---

## 接口一览

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/api/health` | 健康检查（Milvus 地址、collection） |
| POST | `/api/chat` | 流式问答（SSE，事件见下表） |
| POST | `/api/ingest` | 同步入库（小批量，参数 `limit_per_file`） |
| POST | `/api/ingest-stream` | 流式入库（SSE：`start` / `progress` / `done` / `error`） |
| POST | `/api/upload` | 上传文档（.pdf/.docx/.txt/.md/.csv，≤100MB，同名自动改名；按 `content_hash` 去重，重复返回 409） |
| GET | `/api/uploads` | 已上传文档列表 |
| DELETE | `/api/uploads/{doc_id}` | 删除：同时删磁盘文件、数据库记录与已入库向量 |
| GET/POST | `/api/conversations` | 会话列表 / 新建会话 |
| GET/PUT/DELETE | `/api/conversations/{id}` | 会话详情 / 改标题 / 删除（消息级联删除） |
| GET | `/api/conversations/{id}/messages` | 会话消息列表 |
| POST | `/api/feedback` | 提交反馈：`message_id` + `thumbs`(up/down) + 可选 `corrected_answer` / `comment` |
| DELETE | `/api/feedback/{message_id}` | 撤销反馈 |
| GET | `/api/feedback/stats` | 反馈看板：好评/差评/差评率/带纠错数 + 差评清单（反查用户当时的提问） |
| GET | `/api/stats/overview` | 展示大屏总览（单接口返回五屏数据，各板块可独立失败） |

**`POST /api/chat` 的 SSE 事件**

| `type` | 说明 |
|---|---|
| `conversation_id` | 新会话时首条推送 |
| `token` | 答案增量文本（打字机） |
| `rewrite` | 改写后的检索 query |
| `trace` | 检索过程步骤（前端「检索过程」面板，每轮重置） |
| `clarification_request` | 证据不足 → 图已暂停，本轮终点，等用户补充 |
| `correction` | 引用编号被剥除 / 追加了忠实性提示 → 前端**整段替换**回答 |
| `sources` | 引用来源（最后一条） |
| `message_id` | `{id, created_at}` —— 前端拿 id 提交 👍/👎、拿 created_at 显示服务端回复时间 |
| `error` | 服务异常 |

> 流式回答的 `message_id` 是**这一刻**才生成的。缺了它，刚生成的回答就没有 👍/👎 按钮
> （只能刷新页面才点得到），反馈闭环在最后一步断掉。
> ⚠️ 落库**不能一律塞在 `finally`**（客户端断开时 `yield` 会炸）：正常路径先落库再推事件，
> `finally` 只兜异常/断开。

---

## 评估与调优

### 检索质量（`eval_rag.py`）

```bash
python scripts/eval_rag.py build --num 100                     # 从清洗数据抽样建测试集
python scripts/eval_rag.py eval --mode hybrid_rerank           # vector/bm25/hybrid/hybrid_rerank
python scripts/eval_rag.py eval --mode hybrid_rerank --judge   # 额外跑 LLM 裁判打分
python scripts/eval_rag.py eval --mode hybrid_rerank --no-rewrite   # 关闭改写，确定性口径
python scripts/eval_rag.py tune                                # BM25/向量权重网格搜索
python scripts/eval_rag.py compare --mode hybrid_rerank        # 最新一次评估 vs 基线
python scripts/eval_rag.py baseline --mode hybrid_rerank --promote  # 提升当前结果为基线
python scripts/eval_rag.py summary                             # 汇总各模式
```

评估与生产走**同一条** `hybrid_search` 链路，所以 tune 出的最优权重直接落
`backend/data/tuned_weights.json`，后端启动即采纳。

当前基线（`data/eval/baselines/hybrid_rerank.json`）：
`hit_rate 0.92 / mrr 0.9067 / similarity 0.75 / coverage 0.4402 / latency 12.56s`，
`meta.rewrite=false`。选它是因为它是**唯一确定性**的口径（检索路径不含 LLM）。
> 阶段三改改写 Prompt 时曾担心基线作废，但 2026-09-27 全量复跑实测
> **hit_rate / mrr 与基线逐位一致** —— 这个口径本就不调改写，所以基线继续有效。
> 对比时务必报出 `meta.rewrite` 配置差异。

### 对话行为（`eval_dialogue.py`）

对抗样本集 `data/eval/adversarial_set.json`（**严禁取自语料**，改了样本集旧分母全部作废）。

```bash
python scripts/eval_dialogue.py --dry-run              # 秒级：校验样本集 + 规则门离线预判
python scripts/eval_dialogue.py --group followup       # 只跑某分组
python scripts/eval_dialogue.py --diagnose ooc-04,fu-02  # 定点摊开诊断（约 1.5min）
python scripts/eval_dialogue.py                        # 全量（约 15min）
```

三条**硬断言**（进退出码）：① 每轮 trace「意图判定」恰好 1 次；
② 无资料路径（insufficient/chat）回答**不得出现引用编号**；③ insufficient 必须带免责声明。

报告把「没有 `dialogue_act`」（规则门拦下，设计如此）与「分类错」分开；
方向性错误分两档：「医学 → 会话」= 拒答真问题（最危险）、「会话 → 医学」= 白检索一轮。
`--json` 默认落盘 `data/eval/dialogue_results.json`，**读逐条前先核对
`generated_at` / `elapsed_sec` 是否等于本次 stdout 总耗时**。

### badcase 回流（`export_badcases.py`）

```bash
python scripts/export_badcases.py                  # 差评 → data/eval/badcases.jsonl
python scripts/export_badcases.py --include-up     # 连好评一起导（做对照）
```

导出的是**素材**、不是评估集：反馈是主观判断，直接入库等于把「用户当时的口味」
固化成「正确答案」，指标会失真。逐条判断属于「检索错 / 生成错 / 口味问题」后，
前者补 `eval_testset`、中者补对抗集或当标定素材、后者直接丢。

### ⚠️ 关于「可复现」的两条硬结论

1. **`temperature=0` ≠ 可复现**。实测同一 Prompt、同 50 题连算两次，改写完全一致的
   仅 **1/50（2%）**，平均字符相似度 0.638。MoE 路由 / 批大小本身就让贪婪解码不确定。
2. **可复现只能靠「冻结输入」**。`app/rewrite_cache.py` 是唯一来源：命中 ⇒ 逐字一致 ⇒
   指标可比。指纹（Prompt / 模型 / 温度）一变整表作废。
   > 指纹已因阶段三的「焦点：」行刷新：`ff58f0485b2954ca`(105 条) → `20930910b4dc2016`(57 条)。
   > 强制重算：删 `backend/data/cache/rewrite_cache.json`；关闭：`REWRITE_CACHE_ENABLED=false`。

   **做 A/B 时用冻结集**（`REWRITE_CACHE_FROZEN=true`）：它跳过指纹校验、只读加载
   `data/frozen/rewrite_frozen.json`，于是改写 Prompt 怎么改，两边吃的都是同一份改写结果。
   生成/更新快照：`python scripts/freeze_rewrite_cache.py` 然后 `git add -f`。
   ⚠️ 要求**全命中**：跑完看报告里的 `misses`，非 0 就说明冻结集不完整、那几题本轮不可复现。

**做 A/B 前请记住**：`hit_rate` / `mrr` 只依赖检索集，唯一随机源是查询改写。
有改写时 `hit_rate` 的噪声区间约 **0.70~0.82（跨度 6 题）** ⇒ **差距小于 6 题不算证据**；
无改写是确定性的，可照常比。

---

## 测试与自测

离线自测（不需要 Milvus / LLM / MySQL，秒级，全部进 CI）：

```bash
cd backend
python scripts/test_clarify_flow.py     # 阶段二：澄清中断与恢复（含 checkpoint 快照清理）
python scripts/test_focus_entity.py     # 阶段三：跨轮焦点实体
python scripts/test_groundedness.py     # 阶段四：越界引用剥除 + 生成侧自省后处理
python scripts/test_rewrite_cache.py    # 改写缓存：指纹作废 / 冻结模式只读绕过
python scripts/test_history_summary.py  # 长对话历史摘要层
```

> ⚠️ `test_lifespan.py` **不在**上面这组里 —— 它会真连 MySQL / Milvus / LLM 并建表，
> 属于全栈自测（它曾经被误列在离线组里）。

需要全栈的自测：

```bash
python scripts/test_lifespan.py         # 启动钩子（MySQL + Milvus + LLM Key）
python scripts/test_clarify_e2e.py      # 端到端澄清（需后端 + Milvus + LLM Key）
python scripts/test_backend_api.py      # 旧接口回归
python scripts/test_sse_via_proxy.py    # 经 Vite proxy 验证 SSE 流式
```

**改 LangGraph 图后的推荐验证方式**：导入 `app.graph` 后 monkeypatch 掉
`rewrite_for_retrieval` / `get_retriever` / `route_knowledge_source` / `filter_by_source` /
`rerank_documents` 与 `rag_chain` 三条链，用带 `rerank_score` 的假 `Document` 跑 `invoke()`，
断言**节点路径** + **trace 不变量** + rerank 是否被调用。比「起 Milvus 再手测」快两个数量级，
且能覆盖回滚开关。

---

## 运维与排错

**chat 报错排查顺序**

1. `GET /api/health` 是否返回 200；
2. `python scripts/check_data.py` 能否看到数据；
3. 看后端日志里的异常类型：
   - `cannot create index on non-exist field: vector`，或 `not allowed to retrieve raw data of field sparse`
     → 说明有代码又用回了 `langchain-milvus` 的 `Milvus` 封装。该封装默认按 `text/vector/pk`
     命名字段，检索时还会用 `output_fields=["*"]` 取回不可回传的 sparse 字段，与本 schema 不兼容。
     **本项目检索与写入都走 pymilvus 原生 API，不要把封装写法加回来。**
   - `未配置 LLM_API_KEY` → 检查 `backend/.env`。

**重新入库 / 清空数据**

- 重建：`python scripts/ingest.py --cleaned`（先 drop 再建）
- 彻底清空（含数据卷）：`docker compose down -v`
- 删除单条 / 单科室：按 `metadata["department"]` / `metadata["filename"]` 过滤删除

**数据库迁移（Alembic）**

schema 的唯一声明是 `backend/app/models.py`，结构变更走 alembic：

```bash
cd backend
python -m alembic current                              # 当前库处于哪个版本
python -m alembic check                                # ★ 漂移体检：真库 vs models 有无差异
python -m alembic upgrade head                         # 新库：一次建出全部表
python -m alembic stamp head                           # 既有库：只记版本号，不执行 DDL
python -m alembic revision --autogenerate -m "说明"     # 改完 models 后生成迁移
python -m alembic downgrade -1                         # 回退一步
```

当前两条 revision（真库与 `alembic check` 均已零漂移）：

| revision | 作用 |
|---|---|
| `e03c5380d2c8` baseline | 一次建出四张表 + 索引 + 外键。刻意保留真库的**历史形态**：`chat_messages.conversation_id` nullable 且无外键（该表建库早于 `ForeignKey(...)` 声明，`create_all` 不补结构） |
| `0f6e3b0d0396` repair | 收敛到 models 最终形态：**清孤儿 → 收紧 NOT NULL → 补 `fk_chat_messages_conversation_id`**。对空库是空转，对旧库是一步到位 |

> ⚠️ `0f6e3b0d0396` 的 `upgrade` 会**删数据**（清 `conversation_id` 悬空的孤儿消息，
> 实测真库 21 条）。执行前先跑 `python scripts/cleanup_orphan_messages.py`
> 看清单并导出备份（2026-09-27 那次：`data/backup/orphan_messages_20260927_225931.json`）。
> 孤儿在业务上不可达（前端按 `conversation_id` 拉消息），删除不影响任何可见内容。
> `downgrade` 只回退结构，**不还原**被删的行。
> 外键建起后孤儿不会再产生，该脚本从此是「审计 + 备份」工具而非常规清理手段。

> **已有数据的库不要 `upgrade`**（表已存在会报错），用 `stamp head` 接管 ——
> 它只往 `alembic_version` 写版本号，不碰任何表。stamp 之后再用 `upgrade head`
> 补上后续 revision（真库就是这么从 baseline 走到 `0f6e3b0d0396` 的）。
> `alembic/env.py` 直接复用 `app.config.settings`，连接串**不写进** `alembic.ini`
> （一份真相、也不让密码进版本库）。
> ⚠️ `alembic.ini` 必须保持**纯 ASCII**：alembic 用本地码页（中文 Windows 是 GBK）
> 读它，写中文注释会直接 `UnicodeDecodeError: 'gbk' codec`（已踩）。
> 旧库那批手工 DDL（`content_hash`、feedback 唯一约束与复合索引）语义已并入 baseline，
> `scripts/migrate_*.sql` 只作历史记录，新库不再需要执行。

**查看与检索自测**

```bash
python scripts/check_data.py                                  # 概况 + 科室分布 + 样例
python scripts/check_data.py --query "头痛怎么办" --mode hybrid --k 5
```

**前端时间显示口径（不要「统一」）**

- 用户消息用**本地发送时刻**；助手消息用**服务端落库时间**。
  都用本地会让同一个回答的时间在刷新前后跳变。
- 后端返回的是**不带时区的本地串**，`frontend/src/utils/time.ts` 的 `toDate()` 必须先判有无
  时区标记 —— 字段一旦带 `Z`，**不报错、只静默偏 8 小时**。
- 流式期间留空不占位，`patchTimeIfEmpty()` 只兜底、**不覆盖**服务端值。

**前端布局须知**（`.app` 三栏）

`.app` 是 `display:grid; height:100vh; overflow:hidden` ⇒ **必须显式写
`grid-template-rows: minmax(0,1fr)`**。只写 `grid-template-columns` 时隐式行是 `auto`,
会被消息内容撑到几千像素，并引发一串假象（多出大段可滚空白、底部按钮被推出视口、
内部 `overflow-y:auto` 永不触发）。**grid / flex 子项必须显式 `min-height: 0`。**

---

## 医疗安全提示

- 回答**仅供参考**，不能替代执业医师诊断，严禁用于实际诊疗决策
- 数据来自开源医疗问答数据集，可能存在错漏
- 已限制低温度生成、拒答处方剂量，并对急症提示拨打 120
- 接口目前**没有鉴权**，仅适合本机 / 内网 demo

---

## 已知限制

- **证据二档偏粗，`partial` 改走生成侧**：「有主题相关的文档」≠「文档能回答这个问题」。
  主题沾边但答不了时 `top_score` 依然很高（实测 0.576 / 0.914 / 0.647，与真·可回答的
  0.658~1.000 重叠），判 strong 后仍会走 `MEDICAL_PROMPT`，而规则 9 无条件要求标引用 ⇒
  「我不知道」也挂上 `[1][2][3][4][5]`。
  按分数区间切 `partial` 的方案**已实测证伪**，改由**生成侧自省**兜住：
  `detect_unanswerable()` 扫回答开头两句，命中「资料未提供 / 无法回答」这类自陈就
  **整篇剥掉引用编号**（复用阶段四的 `correction` 出口，确定性、可断言、零 LLM）。
  残留局限：措辞是白名单，模型换个说法自认答不上来时仍会漏剥 —— 方向是刻意选的
  （宁可漏剥、不可误剥，误剥会丢掉真回答的出处）。
- **评估不可复现**：`temperature=0` ≠ 可复现，唯一来源是改写缓存；而缓存绑 Prompt
  指纹，指纹一变整表作废、参考点跟着重置。现已支持**冻结集**
  （`REWRITE_CACHE_FROZEN=true`：跳过指纹校验 + 只读不落盘，可 `git add -f` 入库），
  A/B 两边才有同一份输入。详见「评估与调优」的两条硬结论与噪声下限。
- **L2 忠实性核查默认关闭**：先让 L1 跑一段，看 trace 里 unsupported 的分布，
  确认误报率可接受再开。
- **长对话的早期细节仍会丢**：历史摘要层只保留窗口外**用户**发言的前 40 字、最多 4 条；
  更早的助手回答内容不保留（压进来只会挤占窗口）。
