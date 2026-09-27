# 医疗 RAG — 项目长期笔记

> **只留不变量与数字**（目标 <100 行，便于每次快速载入）。机理与逐条实测见同目录的日期日志——每条结论在对应日志里都有完整版。

## 环境
- 后端解释器 `%USERPROFILE%\.conda\envs\RAG\python.exe`（3.11）；backend 下**无** .venv；anaconda base 缺 langchain_huggingface，不是运行环境。
- HF 缓存 `~/.cache/huggingface/hub/`｜MySQL 127.0.0.1:3306｜Milvus 19530（docker `medical-rag-milvus`）｜前端 vite+React18+TS，dev 5173 代理 `/api`→8000。
- LLM 为**硅基流动** OpenAI 兼容接口（`LLM_API_KEY/LLM_BASE_URL/LLM_MODEL`，默认 `deepseek-ai/DeepSeek-V4-Flash`），**并非** DeepSeek 官方接口；`llm.py` 配 `timeout=180, max_retries=1`。`.env.example` 中的变量名已同步为当前版本，旧 `DEEPSEEK_API_KEY` 不再使用。

## 模型离线加载（2026-09-24 落地，注意保持）
根因：**FlagReranker 不吃 `local_files_only`** ⇒ 仍发 HEAD 被 SSL 阻断。四道防线：① `app/__init__.py` 设 `HF_HUB_OFFLINE=1`+`TRANSFORMERS_OFFLINE=1`（离线常量 import 时定型，**必须最早执行**）；② `config.resolve_hf_snapshot()` id→本地快照绝对路径（`refs/main`→`snapshots/` 最新→回退 id，**不硬编码 hash**）；③ `reranker.py` 交绝对路径 ⇒ 走 `os.path.isdir` 分支、零网络；④ PyCharm Run/Debug 同名环境变量兜底。
⚠️ 参数是 `devices`（复数），`device=` 被 kwargs 吞掉、静默失效。

## LangGraph 图（改 `graph.py` 前先读）
`intent ─┬ chat ──→ chat_generate` ｜ `└ medical → 子图[rewrite ─┬ ack/chitchat → END（不检索）│ └ 其余 → route → retrieve → rerank] ─┬ chat → chat_generate ｜ ├ strong → generate → ground_check ｜ └ none ─┬ 可追问且非急症 → human_review（interrupt）｜ └ 否则 → answer_insufficient（禁引用编号）`
- `dialogue_act` 四分类（new_question/followup/ack/chitchat）**由改写那次 LLM 调用顺带产出**（零额外延迟，解析 `query_rewriter._parse_rewrite_output()`）；策略表 `app/dialogue_policy.py`，判据一律走 `policy_for()`。
- 证据只两档：`node_rerank` 清空 `docs` 后 `evidence_state = "strong" if docs else "none"`；`partial` 是空档（分双峰），留位待"覆盖度"定义。
- `MEDICAL_PROMPT` **一个字没动**（规则 9/10 无条件生效）⇒ 无证据路径**整条挪出**该 Prompt，而不是改那两条规则。
- 改写契约：`rewrite_for_retrieval()`→`RewriteResult(query, degraded, reason, transient, dialogue_act)`；`rewrite_query()` 保留 str 签名给 `eval_rag.py`。**D1=A：改写退化时不拼历史主题**——topic 兜底会把「不了」拼成高分命中的病症 query。
- 阶段三 `focus_entity`：**只在 followup 且改写退化时**用焦点补检索词（`"{焦点} {原问题}"`）；两层 = 状态里的显式焦点 → `_focus_from_history(history)`（后者给评测路径，`eval_*` 不带 checkpointer）；`node_requery` 程序化调子图，必须显式传焦点。
- 回滚：`DIALOGUE_ACT_ENABLED`（完全复原要连 Prompt 一起回滚）｜`FOLLOWUP_SCORE_WITH_ENHANCED`｜`REWRITE_GATE_ON_ACT`｜`FOCUS_ENTITY_ENABLED`（依赖前者）。

### ⚠️ 不变量（改动前必读）
1. 「意图判定」每轮 trace **恰好 1 次**：`node_rewrite` **只在真走检索时**写 trace（ack 掉头后与 chat 路径重合）；`eval_dialogue.py` 当硬断言。
2. **ack/chitchat 掉头时不产出 `enhanced_query`**（`astream_answer` 见它就推 `rewrite` 事件）；act 由 chat 节点写进 trace。
3. **父图里排在检索子图之前的节点一律不写 trace**（子图回传 trace 含父图继承部分 + `Annotated[list, add]` ⇒ 重复追加）。`node_classify_intent` 因此不写，由分支后第一个节点 `_intent_step()` 补写。
4. **trace 必须按轮重置**：开 checkpointer 后状态跨轮保留 ⇒ 不重置会滚雪球（第 5 轮 75 条）。由 `node_classify_intent` 发 `TRACE_RESET` 哨兵。
5. **新增会出字的节点必须进 `_astream_run` 的 `STREAMING_NODES`**（现 `generate`/`chat`/`insufficient`），漏了只表现为"没流式效果"。`ground_check` **不得**加入（L2 调 LLM）。
6. 检索子图含 act 分流 ⇒ `rag_chain.retrieve()` 不再"对任何输入都真检索"；**测试集放短寒暄句会让 `hit_rate` 假性下跌**。

### ⚠️ 标签解析 & 流式要点
- `_sanitize_rewrite` 只取第一行 ⇒ **标签行必须先剥再收敛**；`_PREFIX_RE` 不认加粗（实测 `**行为**：followup` 整行会被当检索词送进 Milvus），靠 `_ACT_LINE_RE`/`_QUERY_LINE_RE`（容纳 `*`）兜并直接解析「查询：」行的取值；`_LABEL_RESIDUE_RE` 判失败时**回退原问题**（宁可少赚，不可投毒）。
- **代码追加到 `answer` 的文本会随流式路径静默丢失** ⇒ 节点须额外返回 `answer_suffix`，由 `astream_answer` 在 token 流结束后补发（只在 `emitted_tokens=True`）。
- 阶段四 `correction` 事件：`main.py` 必须消费并**覆盖 `full_answer`**，否则页面显示修正版、库里存原始版（刷新后已剥除的越界编号会重新出现）。

### rerank 用哪个 query 打分（已定）
按 act 分流：**followup 用 `enhanced_query`，其余用原始 `question`**。两条必须同时成立：原句才是「不了」被挡住的真正原因（用改写打分会让它高分命中病症资料）；followup 必须用改写，因为真·指代追问按原句打分极低（「它有什么副作用」=0.355，仅高于阈值 0.055）→ 被兜底成"没资料"。trace 的「精排」步带 `scored_by`（`原句`/`改写 query`）。

## 评估复现性与改写效果（做 A/B 前必读）
`hit_rate`/`mrr` **只依赖检索集**（判标准答案前 80 字是否出现在召回里，judge=false 不调 LLM），检索集只由 `enhanced_query` 决定 ⇒ **唯一随机源是查询改写**。
- **`temperature=0` ≠ 可复现**：同 Prompt 同题连算两次，改写完全一致仅 **1/50 = 2%**，平均字符相似度 **0.638**。MoE 路由/批大小本身就让贪婪解码不确定。**因此不能以「跨进程结果一致」推断可复现——那只说明缓存在生效。**
- **可复现只能靠「冻结输入」**：`app/rewrite_cache.py` 是**唯一**来源（命中⇒逐字一致⇒指标可比），指纹一变（Prompt/模型/温度）整表作废。⚠️ 指纹已因阶段三「焦点：」行刷新：`ff58f0485b2954ca`(105) → `20930910b4dc2016`(57)。**09-27 实测**：新缓存 = 对抗集（应覆盖 **54/54 全命中**；缺的 11 条=规则门拦下、本不调 LLM），但 **`eval_testset` 50 题 0 覆盖** ⇒ 有改写口径当前不可复现，需先重灌缓存。⚠️ `backend/data/` 在 `.gitignore` ⇒ 冻结内容未进版本库，尚缺「入库快照 + 绕过指纹的加载开关」——**尚未实现**。强制重算删 `backend/data/cache/rewrite_cache.json`；关闭 `REWRITE_CACHE_ENABLED=false`。
- **改写在这份测试集上未带来提升**（同代码、测试集 md5 `ed45e0e2`）：**无改写** hit_rate **0.92** / mrr **0.9067**（确定性）｜有改写抽样 0.82 / 0.70（mrr 0.7933 / 0.6867，不可复现）。逐题 12 题差异 → **6 题「无改写对、改写两次都错」，0 题反向**（符号检验 p≈0.016）。机制：【改写要求】自相矛盾（规则 2 要补同义词、规则 4 又不引入新病症）⇒ 模型倾向输出关键词堆、带入"鉴别诊断/肋骨骨折"这类**用户未提及且未必准确**的词。
- **噪声下限**：有改写时 `hit_rate` ∈ **0.70~0.82**（跨度 6 题）⇒ **差距小于 6 题不足以作为证据**；无改写是确定性的，可照常比较。

## 对抗评测（样本集 65 条/9 组，严禁取自语料）
**09-27 全量复跑（最新分母）**：act **54/54=100%**、落点 **59/62=95.2%**、**方向性错误 0 条 ✅**、引用一致性 ✅、规则门拦 11 条**零误伤**、17.2s/条、缓存 **54 hits/0 misses（可复现）**。落点不符 3 条均为「裸症状主诉/库外被判有资料」：`ooc-05`(0.330)、`sm-08`「好痛」(0.501)、`gm-03`「好难受」(0.523) → **阈值问题，非分类问题**。⚠️ 旧分母 55/55、59/61 **已不适用**（样本集与规则门有调整）。

### 需注意：`evidence=strong` 不等同于「可回答」（= 阶段二入口）
**「有主题相关的文档」≠「文档能回答这个问题」**：ooc-04（0.576）/ooc-08（0.914）/ooc-09（0.647）都是"库里沾边但答不了" → `top_score` 高 → 判 strong → 走 `MEDICAL_PROMPT` → 规则 9 无条件要求标引用 ⇒ 给"我不知道"挂上 `[1]..[5]`——编号真实存在、**不算伪造出处**，但让"答不上来"看着有出处（**与最初那个 bug 同源**）。
标定（实测）：真·可回答 **0.658~1.000**｜主题相关但不可回答 **0.576/0.914/0.647**｜真·库外 **0.010~0.137** ⇒ **单靠 `top_score` 无法区分**，"partial 按区间切"的方案**已排除**。候选信号：覆盖度 / **生成侧自省后处理**（回答出现「资料未提供/无法回答/仅提及」就剥引用编号，确定性可断言）。

### ✅ 规则门（`app/intent.py`）四条要点
1. **假阳性影响最大（会拒答真问题）**：`_is_pure_ack()` 要求整串能被「确认词+语气助词」**吃干净**（前往后贪心，优先吃确认词）。只"从尾部剥语气词"不够（「行，知道了」剥掉「了」→「知道」不在词表→吃不完）。**整串字面相等必须排在线索检查之前**，否则「麻烦你了」含「麻」（发麻线索）会被永久短路。子串匹配+单字「好」入表曾让「好恶心」「好困」被判为会话语，**第二层 LLM 未被调用**。
2. **假阴性会静默走错分支**：`不了` 曾漏（同族只收「没了/不用了/不需要」），生产数据抓到过现行：判 False→检索并生成带 `[1]` 的科普，**答非所问且表面无异常**。⇒ 补词表**按"同族穷举"过一遍**；**不能**把「你」塞进 `_TAIL_CHARS`。`_MEDICAL_HINTS` 已补「恶心/困/难受/胀」兜底。
3. **同一缺陷会在第二层复现**：「好困」规则门放行后**仍被 LLM 判 chitchat**——两份改写 Prompt 里 chitchat 例子**全是问候**、ack 又把「好的」列进去，**判据空白**。修法：`REWRITE_PROMPT` 与 `CONTEXT_REWRITE_PROMPT` **同时**追加【症状主诉不是闲聊】。
4. **改 Prompt ⇒ 缓存指纹整体作废**，基线随之失效，必须重跑 `eval_rag`；**修完一层必须再跑全量**。**判据刻意偏保守**（「我明白了」「改天聊，拜拜」落到第二层）——代价不对称，第一层假阳性会导致直接拒答且不可恢复。
- **生成侧同样存在波动**：`ooc-08` 引用编号两次运行不一致而检索集完全相同 ⇒ 差异来自 `temperature=0.3`。**涉及生成文本的断言不宜用单次运行下结论。**

## 基线
⚠️ 阶段三改改写 Prompt 时曾担心基线失效——**09-27 全量复跑实测：hit_rate 0.92 / mrr 0.9067 与基线逐位一致**（no-rewrite 口径不调改写，本就不受影响）⇒ **基线继续有效，无需重冻结**。coverage/similarity 的波动属生成侧噪声（`temperature=0.3`）、latency 含首载开销，**不宜当作回归**。
`data/eval/baselines/hybrid_rerank.json` = **hit_rate 0.92 / mrr 0.9067 / similarity 0.75 / coverage 0.4402 / latency 12.56s，`meta.rewrite=false`**。取它是因为**唯一确定性**；旧基线 0.82 的改写集随指纹作废、**不宜作为参考点**；对比时务必报出 `meta.rewrite` 差异。换回"有改写为准"：`eval --mode hybrid_rerank` 后 `baseline --mode hybrid_rerank --promote`。

## 反馈闭环 & 前端溯源
- `GET /api/feedback/stats`：好评/差评/差评率/带纠错数 + 差评清单（**反查"用户当时问的是什么"**）。造测试数据只 `flush` 不 `commit`。
- ✅ **`message_id` 必须回传**（否则刚生成的回答没有 👍/👎）：`_persist_assistant_message` 返回 `(id, created_at)` → 正常收尾**先落库再推** `message_id` → `onMessageId` 挂到消息上。⚠️ **落库不能一律留在 `finally`**（客户端断开时 yield 会抛异常）：用 `saved` 标志，`finally` 只兜异常/断开。
- 反馈交互：👍 直提；👎 走 `openDownForm()` 表单，`corrected_answer` 与 `comment` **都有录入入口**，`withdrawFeedback()` 可撤销。两处已知待改进：失败仅 `console.error` 未提示、看板清单只列差评。
- 引用溯源：`renderAnswer()` 把 `[n]` 渲染成可点击锚点 → 滚到来源卡片 + 高亮 1.6s；卡片 id=`cite-{消息下标}-{编号}`；**只对 `sources` 里真实存在的编号生效**；正文容器 `white-space: pre-wrap`。

## 前端（改 `.app` 三栏或时间显示前必读）
- `.app` 是 `display:grid; height:100vh; overflow:hidden` ⇒ **必须显式写 `grid-template-rows: minmax(0,1fr)`**。只写 `grid-template-columns` 时隐式行是 `auto`、被消息撑到几千像素（实测 3053px），引发一串连锁现象：多出 ~3000px 可滚空白、`.rail-spacer` 吸走多余高度把底部按钮推出视口、`overflow-y:auto` **永不触发**。**grid/flex 子项必须显式 `min-height: 0`**。另：`overflow:hidden` 的元素**是**滚动容器、会被 `scrollIntoView` 滚动。
- 断点：`max-width:1180px`（证据栏变抽屉）/`860px`（图标栏变抽屉）/`max-height:900px`（海报 `.intro-foot` 改 sticky，否则 1366×768 下「进入工作台」在折叠线以下）。
- **消息时间**：用户消息用**本地发送时刻**，助手消息用**服务端落库时间**——口径不同，需分别处理。⚠️ 后端是"不带时区的本地串"，`toDate()` 必须先判有无时区标记——字段一旦带 `Z`，**不报错、只静默偏 8 小时**。流式期间留空不占位，`patchTimeIfEmpty()` 只兜底、**不覆盖**服务端值。
- 会话选中态落盘 `localStorage['medical-rag.active-conversation-id']`（后端无此状态）；主题在模块作用域写 `<html>`（放 useEffect 会闪白）。

## 工具使用方式
- **`scripts/eval_dialogue.py`**：`--dry-run` 秒级（校验样本集 + `gate_preview()` 纯规则零 LLM，**改 `intent.py` 前先扫、改完再验**）；`--diagnose id1,id2` 定点摊开 `enhanced_query`/`scored_by`/`top_score`/`sources`/回答尾/引用一致性（~1.5min vs 全量 ~15min）。
- **三条硬断言**（进退出码）：① 每轮 trace「意图判定」恰好 1 次；② 无资料路径回答**不得出现引用编号**；③ insufficient 必须带免责声明。
- `--json` 默认落盘 `data/eval/dialogue_results.json`，⚠️ 读逐条前**先核对 `generated_at`/`elapsed_sec`** 是否等于本次 stdout 总耗时（曾因默认 None 拿到上一轮旧文件）。
- `run_item` 的 `enhanced_query`/`scored_by`/`sources_head`/`answer_tail`/`has_citation`/`has_disclaimer` 是**诊断必需字段，勿删**。报告把"没有 `dialogue_act`"（规则门拦下，设计如此）与"分类错"分开；方向性错误分两档：「医学→会话」=拒答真问题（影响最大）、「会话→医学」=多一次无效检索。`hit_rate` **测不出**分类好坏。
- **改 LangGraph 图后的离线验证**（不依赖 Milvus/LLM/权重）：导入 `app.graph` 后 monkeypatch 掉 `rewrite_for_retrieval`/`get_retriever`/`route_knowledge_source`/`filter_by_source`/`rerank_documents` 与 `app.rag_chain` 三条链，用假 `Document`（带 `rerank_score`）跑 `invoke()`，断言**节点路径** + **trace 不变量** + rerank 是否被调用。比"起 Milvus 再手测"快两个数量级，且能覆盖回滚开关。
- **离线自测**（不需 Milvus/LLM/MySQL）：`test_clarify_flow.py` / `test_focus_entity.py` / `test_groundedness.py` / `test_lifespan.py`。
- **前端改动后的验证**：用 `browser-verify-via-cdp` 技能（本机 Chrome+CDP）；后端需 `uvicorn app.main:app --port 8000`；脚本写 `%TEMP%`、跑完清理。**断言用计算样式/`getBoundingClientRect`，不靠"截图看起来对"。**

## 已知约定
- 检索链路 top_k 一律走 `settings.reranker_top_k`，不硬编码；检索逻辑只在 `backend/app/graph.py` 维护一份，`rag_chain` 委托它。
- `backend/.env` 曾出现残行，注意别写坏。
