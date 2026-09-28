# 医疗 RAG — 项目长期笔记

> 只留不变量与数字；机理与逐条实测见同目录日期日志（每条结论在对应日志里都有完整版）。

## 环境
- 后端：`%USERPROFILE%\.conda\envs\RAG\python.exe`（3.11；backend 下无 .venv；anaconda base 不是运行环境）。
- 前端 vite+React18+TS，dev 5173 代理 `/api`→8000｜MySQL 127.0.0.1:3306｜Milvus 19530（docker `medical-rag-milvus`）｜HF 缓存 `~/.cache/huggingface/hub/`。
- LLM = **硅基流动** OpenAI 兼容接口（`LLM_API_KEY/LLM_BASE_URL/LLM_MODEL`，默认 `deepseek-ai/DeepSeek-V4-Flash`），**不是** DeepSeek 官方；`llm.py` 配 timeout=180 / max_retries=1。

## 模型离线加载（勿破坏）
根因：`FlagReranker` 不吃 `local_files_only` ⇒ 仍发 HEAD 被 SSL 阻断。四道防线：① `app/__init__.py` **最早**设 `HF_HUB_OFFLINE=1`+`TRANSFORMERS_OFFLINE=1`；② `config.resolve_hf_snapshot()` id→本地快照绝对路径（不硬编码 hash）；③ `reranker.py` 收绝对路径 ⇒ 走 `os.path.isdir` 零网络；④ PyCharm 环境变量兜底。
⚠️ 参数是 `devices`（复数），`device=` 会被 kwargs 吞掉、静默失效。

## LangGraph 图（改 `graph.py` 前先读）
`intent ─┬ chat → chat_generate` ｜ `└ medical → 子图[rewrite ─┬ ack/chitchat → END（不检索）│ └ 其余 → route → retrieve → rerank] ─┬ chat → chat_generate │ ├ strong → generate → ground_check │ └ none ─┬ 可追问且非急症 → human_review（interrupt）│ └ 否则 → answer_insufficient（禁引用编号）`
- `dialogue_act` 四分类由改写那次 LLM 调用顺带产出（零额外延迟）；策略表 `app/dialogue_policy.py`，判据一律走 `policy_for()`。
- 证据只两档：strong / none（`partial` 是空档留位，分双峰）。
- `MEDICAL_PROMPT` **一个字没动**（规则 9/10 无条件生效）⇒ 无证据路径整条挪出该 Prompt。
- 改写契约 `rewrite_for_retrieval()` → `RewriteResult(query, degraded, reason, transient, dialogue_act)`；`rewrite_query()` 保留 str 签名给 `eval_rag.py`。**D1=A：改写退化时不拼历史主题**。
- 阶段三 `focus_entity`：只在 followup 且改写退化时用焦点补检索词；两层 = 状态显式焦点 → `_focus_from_history(history)`（评测路径用）。`node_requery` 程序化调子图须显式传焦点。
- 回滚开关：`DIALOGUE_ACT_ENABLED`｜`FOLLOWUP_SCORE_WITH_ENHANCED`｜`REWRITE_GATE_ON_ACT`｜`FOCUS_ENTITY_ENABLED`。

### ⚠️ 不变量
1. 每轮 trace「意图判定」**恰好 1 次**（`node_rewrite` 只在真走检索时写；`eval_dialogue.py` 当硬断言）。
2. ack/chitchat 掉头时**不产出** `enhanced_query`（`astream_answer` 见它就推 `rewrite` 事件）。
3. 父图中排在检索子图之前的节点一律不写 trace（`Annotated[list, add]` ⇒ 重复追加）。
4. trace 必须按轮重置（由 `node_classify_intent` 发 `TRACE_RESET` 哨兵）。
5. 新增会出字的节点必须进 `_astream_run` 的 `STREAMING_NODES`（现 generate/chat/insufficient）；`ground_check` **不得**加入。
6. 检索子图含 act 分流 ⇒ 测试集放短寒暄句会让 `hit_rate` 假性下跌。

### ⚠️ 标签解析 & 流式
- `_sanitize_rewrite` 只取第一行 ⇒ 标签行必须先剥再收敛；`_PREFIX_RE` 不认加粗，靠 `_ACT_LINE_RE`/`_QUERY_LINE_RE` 兜；解析失败回退原问题。
- 代码追加到 `answer` 的文本会随流式静默丢失 ⇒ 节点须额外返回 `answer_suffix`，由 `astream_answer` 在 token 流后补发。
- 阶段四 `correction` 事件：`main.py` 必须消费并**覆盖 `full_answer`**。

### rerank 打分用哪个 query（已定）
**followup 用 `enhanced_query`，其余用原始 `question`**。trace「精排」步带 `scored_by`。

## 评估口径（做 A/B 前必读）
- `hit_rate`/`mrr` **只依赖检索集** ⇒ 唯一随机源是查询改写。
- **`temperature=0` ≠ 可复现**：同 Prompt 连算两次改写完全一致仅 **1/50**，平均相似度 **0.638**。可复现只能靠冻结输入：`app/rewrite_cache.py` 是唯一来源，指纹一变（Prompt/模型/温度）整表作废（当前 `20930910b4dc2016`）。`REWRITE_CACHE_FROZEN=true` ⇒ 跳指纹校验 + 只读不落盘。
- 09-27 实测：对抗集 54/54 全命中；但 `eval_testset` 50 题 **0 覆盖** ⇒ 有改写口径当前不可复现。冻结集已用 `git add -f` 入库（`backend/data/` 整目录被 ignore，不 `-f` 进不去），见 `backend/data/frozen/rewrite_frozen.json`。
- **改写在这份测试集上未带来提升**（测试集 md5 `ed45e0e2`）：无改写 **0.92 / mrr 0.9067**（确定性）｜有改写 0.82 / 0.70（不可复现）。逐题 12 题差异 → 6 题「无改写对、改写两次都错」、0 题反向（符号检验 p≈0.016）。机制：【改写要求】自相矛盾（规则 2 补同义词 vs 规则 4 不引入新病症）⇒ 输出关键词堆、带入用户未提及的词。
- **噪声下限**：有改写 hit_rate ∈ **0.70~0.82** ⇒ 差距 **<6 题不足以作证据**；无改写是确定性的，可照常比较。

## 对抗评测（样本集 **71 条/10 组**，严禁取自语料）
- **09-28 新增 `focus_entity` 组 6 条**（`fx-01`~`fx-05` 为 `observe: true`、`fx-06` 是反向样本 `好的`→ack/chat）：`observe` 只判 act、**不进落点分母**。冻结集随之 **57→66 条**（原缓存的子集，只补 9 条、指纹未变）。
- 规则门预判（`--dry-run`，纯规则零 LLM）：拦 **12/71 条、零误伤**，`fx-06` 被正确拦下 ⇒ 两层分类器没打架。
- **09-28 全量复跑（71 条口径，当前基线）**：act **59/59 = 100%**、落点 **60/63 = 95.2%**（8 条 observe 不计入）、**方向性错误 0** ✅、规则门拦 **12/71 零误伤**、**13.0s/条**、缓存 `size 66 / hits 59 / **misses 0**`、退出码 0。焦点组 act 5/5（top 0.958~0.999）。
  ⚠️ **该轮跑了两次，数字逐位一致** ⇒ ① 冻结缓存确实买到了可复现性；② 这组数字可当基线。落点不符仍是同样 3 条（`ooc-05` 0.330 / `sm-08` 0.501 / `gm-03` 0.523）= 阈值问题。
- **09-28 晚复跑（B-1/B-3 之后）**：同为 `59/59 · 60/63 · 12/71 · 缓存 66/59/0`，且与 B-1 前那轮**逐条字段级 0 差异**（71 条 id 集合相同；`act/route/gated/evidence/n_sources/top_score/enhanced_query/scored_by/steps` 全同）⇒ 砍回边 + 看门狗对默认路径**零副作用**。⚠️ 两轮 elapsed 差一倍（1236.8s → 622.8s）但判定全同 ⇒ **耗时是 provider 噪声、不是回归指标**；可靠回归 = `--json` 落盘后按字段比对（旧 json 多 `grade_retry` 字段属预期，已删）。
- ⚠️ **分母随样本集变，不能混用**：65 条口径 act 54/54、落点 59/62；71 条口径 act 59/59、落点 60/63。旧数字留档见下。
09-27 全量复跑（65 条口径，留档）：act **54/54**、落点 **59/62**、**方向性错误 0** ✅、规则门拦 11 条**零误伤**、17.2s/条、缓存 54 hits/0 misses。落点不符 3 条均为「裸症状主诉/库外被判有资料」：`ooc-05`(0.330)、`sm-08`「好痛」(0.501)、`gm-03`「好难受」(0.523) ⇒ **阈值问题，非分类问题**。
- **`evidence=strong` ≠ 「可回答」**（= 阶段二入口）：主题沾边但答不了（0.576/0.914/0.647）与真·可回答（0.658~1.000）**`top_score` 分不开** ⇒「partial 按区间切」已排除，改走生成侧自省（T59）。
- **规则门（`app/intent.py`）**：① 假阳性最致命（会拒答真问题）——`_is_pure_ack()` 要求整串被"确认词+语气助词"吃干净；**整串字面相等必须排在线索检查之前**；子串匹配 + 单字「好」入表曾让「好恶心/好困」被误判。② 假阴性会静默走错分支（`不了` 曾漏）⇒ 补词表按"同族穷举"过一遍；不能把「你」塞进 `_TAIL_CHARS`。③ 同一缺陷会在第二层复现 ⇒ `REWRITE_PROMPT` 与 `CONTEXT_REWRITE_PROMPT` **同时**追加【症状主诉不是闲聊】。④ **改 Prompt ⇒ 缓存指纹整体作废**，必须重跑全量。
- **生成侧同样波动**（`ooc-08` 两次引用编号不同而检索集相同，源自 `temperature=0.3`）⇒ 涉生成文本的断言别用单次运行下结论。

## 工程化基座（2026-09-27 落地）
- **Alembic**：`alembic check` 是漂移体检唯一权威。`e03c5380d2c8` baseline（**刻意保留真库历史形态**：conversation_id nullable + 无外键）→ `0f6e3b0d0396` repair（清孤儿 → 收紧 NOT NULL → 补 `fk_chat_messages_conversation_id`）。⚠️ **不能把 baseline 直接写成最终形态**（真库 stamp 在 head 后缺失的外键永远补不上）。真库现状 = 202 条消息 / 孤儿 0 / stamp `0f6e3b0d0396`。`alembic.ini` **必须纯 ASCII**（GBK 读、中文注释直接 `UnicodeDecodeError`）；连接串只在 `env.py` 取；`%` 写 `%%`；约束名靠 `naming_convention`（**别带 `%(referred_table_name)s`**）。
- **T59 走生成侧自省**：`detect_unanswerable()` 扫回答**开头两句**命中白名单 ⇒ `strip_all_citations()` 整篇剥编号（复用 `correction` 事件下发给前端覆盖）。
- **T61 历史摘要**：`format_history()` 超 6 轮折叠更早**用户**发言（4 条 × 40 字）。⚠️ 开关判断必须在 `len(history) <= max_turns` 提前返回**之后**。
- **CI** `.github/workflows/ci.yml` 双 job：离线 **7** 项测试 + `ruff --select E9,F backend/app`。离线 7 项 = clarify_flow / focus_entity / groundedness / rewrite_cache / history_summary / node_reliability / sse_idle（**`test_lifespan` 不能进**）。`scripts/` **不进 lint**。CI 需先预装 CPU 版 torch。
- **进度表**：`_build_progress.py`（62 卡 / 11 阶段）需 **openpyxl**，用 `~\.workbuddy\binaries\python\versions\3.13.12\python.exe` 跑（RAG conda env 里没有）。现况：已测试 **58/62**（口径A 93.5%）｜交付口径(P0~P9) **53/55**（96.4%）｜未开发 2 = T55/T57（T62 已开发：⑧①④ 落地 + B-1/B-3）。
- **T62-B 收口（2026-09-28 晚）**：① **④ 回边整条删除**（`_same_query` 守卫本身是死代码——键在 `verdict=="retry"` 而 `parse_grade_output` 只认字面 `insufficient`；净收益全在「判不足→走无资料路径」半边）；连带清掉 `grade_retry` 字段 / `_route_after_grade` / `node_rewrite` 直通分支 / `EVIDENCE_GRADE_MAX_RETRIES` / Prompt 的 `retry_query`（改 2 键），`parse_grade_output` 改**2 元组**。② **流式空闲看门狗**：后端 `main.py` 的 `_watchdog_iter`（`SSE_IDLE_TIMEOUT_SEC` 默认 90s，超时补兜底文案 + 推 `correction`/`verdict=stream_idle_timeout`）｜前端 `Chat.tsx` 整轮 120s → **空闲 105s**（>后端 90s 让后端先收尾）。**这才是真修法**：`sse_starlette` 每 15s 发 `: ping`，整轮计时会误杀仍在生成的流。③ 新增 `test_sse_idle.py`（7 组；护栏 B3-3 = 总耗时>idle 但每间隔<idle 一条不丢）。`node_reliability` 116 项 PASS。④ **`EVIDENCE_GRADE_ENABLED` 保持默认关**（已拍板）：每轮 +1 次 LLM 调用、尾延迟下墙钟不可控；要开先降 `LLM_TIMEOUT_SEC`/`NODE_RETRY_MAX_ATTEMPTS`。

## 端到端验收（T49，2026-09-28 完成）
- **手段**：本机 Chrome + CDP 无头驱动（`~/.workbuddy/skills/browser-verify-via-cdp`），脚本写 `%TEMP%` 用完即删。关键技巧：① 断言读 `.trace-detail` **文本**，不读截图；② 「复制到的是不是修正版」= **劫持 `navigator.clipboard.writeText` 记录真实载荷**（读到即 `m.content` 原文），别用 `readText()`（headless 要授权）；③ 受控输入走 `HTMLTextAreaElement.prototype` 原生 setter + `input` 事件 + `form.requestSubmit()`；④ 开新会话 = 清 `localStorage['medical-rag.active-conversation-id']` + `Page.reload`；⑤ 越界编号用**环境变量门控的临时注入**构造，验完 `git checkout backend/app/graph.py`（`git diff --quiet` 确认）。
- **结果**：阶段三 §3.5 四项 ✅｜阶段四 §3.4 五项 ✅｜阶段二 #3 / #9 ✅｜异常 0 / console.error 0。**命门（刷新后仍是修正版）实测通过**：注入 `[9]` → trace `verdict: citation_fixed · invalid: 9` → 页面与复制载荷均无 `[9]` → 刷新前后原文**逐字一致**。
- **配置注入优先用环境变量，不改 `.env`**：pydantic-settings 环境变量优先于 `.env`，而 `backend/.env` 是 **CRLF**，混入 LF 行有留残行风险。`GROUNDEDNESS_LLM_ENABLED=true` / `HUMAN_REVIEW_ENABLED=false` 都是这样临时注入的。
- **⚠️ provider 尾延迟是当前最大噪声源**：同一句话连算 3 次 = **1.4s / 33.6s / 254.1s**（254s = 180s 超时 + 重试成功）。而前端 `Chat.tsx` 判据是**整轮墙钟 120s**（`AbortController`），后端单轮最坏 = 2 次 LLM 调用 ×（`timeout=180` × `max_retries=1`）= **720s** ⇒ 口径差一个数量级，表现为「响应流已中断，未收到任何内容」+ 后端「客户端断开连接 / 回答为空跳过落库」。`sse_starlette 2.1.0` 默认每 **15s** 发 `: ping`，连接其实一直是活的 ⇒ **修法是前端改 idle 超时（已落地：前端 105s + 后端 90s 兜底）**，不是加长整轮超时。已写进 README「已知限制」。
- **`eval_dialogue.py` 已加单条隔离**：一条 provider 卡顿曾让 67/71 处崩掉、整轮 30 分钟与已跑完的 66 条一起作废。现在 `run_item` 外包 try/except → `_failed_result()` 占位，失败条目**不进任何分母**、单列计数、**强制退出码 1**（`route_n` 已扣除、`verdict_fail` 含 `n_error`）。离线验证脚本 7/7 通过（合成结果直接喂 `report()`）。

## 基线
`data/eval/baselines/hybrid_rerank.json` = **hit_rate 0.92 / mrr 0.9067 / similarity 0.75 / coverage 0.4402 / latency 12.56s，`meta.rewrite=false`**。取它因为**唯一确定性**；09-27 全量复跑逐位一致 ⇒ 基线继续有效、无需重冻结。coverage/similarity 波动属生成侧噪声、latency 含首载开销，**不宜当作回归**。对比时务必报出 `meta.rewrite` 差异。换回"有改写为准"：`eval --mode hybrid_rerank` 后 `baseline --mode hybrid_rerank --promote`。

## 反馈闭环 & 前端溯源
- `GET /api/feedback/stats`：好评/差评/差评率/带纠错数 + 差评清单。造测试数据只 `flush` 不 `commit`。
- **`message_id` 必须回传**（否则刚生成的回答没有 👍/👎）：先落库再推 `message_id`。⚠️ 落库**不能一律留在 `finally`**（客户端断开时 yield 抛异常）——用 `saved` 标志。
- 反馈交互：👍 直提；👎 走 `openDownForm()`（`corrected_answer` 与 `comment` 都有入口），`withdrawFeedback()` 可撤销。已知待改进：失败仅 `console.error` 未提示、看板只列差评。
- 引用溯源：`renderAnswer()` 把 `[n]` 渲染成锚点 → 滚到来源卡片 + 高亮 1.6s；**只对 `sources` 里真实存在的编号生效**；正文容器 `white-space: pre-wrap`。

## 前端（改 `.app` 三栏或时间显示前必读）
- `.app` 是 `display:grid; height:100vh; overflow:hidden` ⇒ **必须显式写 `grid-template-rows: minmax(0,1fr)`**（否则隐式行 auto、被消息撑到几千像素，引发可滚空白 + `.rail-spacer` 把按钮推出视口 + `overflow-y:auto` 永不触发）。**grid/flex 子项必须显式 `min-height: 0`**。`overflow:hidden` 的元素**是**滚动容器。
- 断点：1180px（证据栏变抽屉）/ 860px（图标栏变抽屉）/ max-height:900px（海报 `.intro-foot` 改 sticky）。
- **消息时间**：用户用本地发送时刻、助手用服务端落库时间。⚠️ 后端是"不带时区的本地串"，`toDate()` 必须先判有无时区标记——字段一旦带 `Z`，不报错、只静默偏 8 小时。`patchTimeIfEmpty()` 只兜底不覆盖。
- 会话选中态落盘 `localStorage['medical-rag.active-conversation-id']`；主题在模块作用域写 `<html>`（放 useEffect 会闪白）。

## 工具使用方式
- `scripts/eval_dialogue.py`：`--dry-run` 秒级（校验样本集 + `gate_preview()` 纯规则零 LLM，**改 `intent.py` 前先扫、改完再验**）；`--diagnose id1,id2` 定点摊开（~1.5min vs 全量 ~15min）。
- **三条硬断言**（进退出码）：① 每轮 trace「意图判定」恰好 1 次；② 无资料路径回答不得出现引用编号；③ insufficient 必须带免责声明。
- `--json` 默认落 `data/eval/dialogue_results.json`，⚠️ 读逐条前先核对 `generated_at`/`elapsed_sec`。
- `run_item` 的 `enhanced_query`/`scored_by`/`sources_head`/`answer_tail`/`has_citation`/`has_disclaimer` 是**诊断必需字段，勿删**。
- **改图后的离线验证**（不依赖 Milvus/LLM/权重）：monkeypatch 掉 `rewrite_for_retrieval`/`get_retriever`/`route_knowledge_source`/`filter_by_source`/`rerank_documents` 与 `app.rag_chain` 三条链，用假 `Document` 跑 `invoke()`，断言节点路径 + trace 不变量 + rerank 是否被调用。
- **离线自测**（不需 Milvus/LLM/MySQL）：`test_clarify_flow` / `test_focus_entity` / `test_groundedness` / `test_lifespan`。
- **前端验证**：`browser-verify-via-cdp` 技能（本机 Chrome+CDP）；脚本写 `%TEMP%`、跑完清理；断言用计算样式/`getBoundingClientRect`。

## 已知约定
- 检索链路 top_k 一律走 `settings.reranker_top_k`，不硬编码；检索逻辑只在 `backend/app/graph.py` 维护一份，`rag_chain` 委托它。
- `backend/.env` 曾出现残行，注意别写坏。
