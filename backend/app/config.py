"""全局配置：从 .env 读取，pydantic-settings 管理类型与默认值"""
import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    #上传文档
    upload_dir: str = "uploads"  # 上传文件目录 (相对 backend)
    upload_max_size_mb: int = 100  # 单文件大小上限
    upload_allowed_extensions: str = ".pdf,.docx,.txt,.md,.csv"  # 允许的扩展名

    # LLM（硅基流动 OpenAI 兼容接口）
    llm_api_key: str = ""
    llm_base_url: str = "https://api.siliconflow.cn/v1"
    llm_model: str = "deepseek-ai/DeepSeek-V4-Flash"

    # Milvus
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection: str = "medical_qa"

    # Embedding
    embedding_model: str = "BAAI/bge-large-zh-v1.5"
    embedding_device: str = "cpu"

    # Reranker（模型 id 会经 resolve_hf_snapshot 解析成本地快照绝对路径，见文件末尾）
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_device: str = "cuda"

    #查询改写
    query_rewrite_enabled: bool = True
    # 改写结果长度上限（字符）。超过即判定"LLM 返回的是解释而非查询"，回退原问题。
    # 定 120 而非 40：正常改写产物（关键词堆叠）实测 68 字，卡太紧会误杀。
    rewrite_max_chars: int = 120

    # ---- 改写可复现（评估可比性的前提）----
    # 改写用的采样温度。关键词抽取这类任务应零温——0.3 会让同一问题每次改写不同，
    # 直接导致评估指标不可复现（实测同一份代码两次跑 hit_rate 差 4 题）。
    rewrite_temperature: float = 0.0
    # 改写结果落盘复用：跨进程、跨运行拿到同一个 query，评估才可比
    rewrite_cache_enabled: bool = True
    rewrite_cache_path: str = "data/cache/rewrite_cache.json"
    # 缓存条目上限，超出按插入顺序淘汰最旧的（防文件无限膨胀）
    rewrite_cache_max_entries: int = 5000
    # ---- 长对话历史摘要（T61）----
    # 历史超出窗口（format_history 的 max_turns）时，把更早的用户发言压成一行要点，
    # 避免"第 1 轮说的病情到第 12 轮已不在上下文里"。纯确定性、零 LLM 调用。
    # 关掉 = 退回"只取最近 N 条"。注意它只在真的截断时才有影响，
    # 因此对短对话的评估口径（含全部对抗集样本）没有任何改变。
    history_summary_enabled: bool = True
    # ---- 冻结改写集（A/B 可比的强约束）----
    # 把某次运行冻结下来的改写结果当**只读输入**加载：跳过指纹校验、命中即逐字复用。
    # 于是即使后来改了 Prompt（指纹变了），A/B 两边仍吃同一份改写结果。
    # ⚠️ 要求**全命中**：miss 会真调 LLM 且不落盘 ⇒ 那几题本轮不可复现（看 stats.misses）。
    rewrite_cache_frozen: bool = False
    # 冻结集文件路径（独立于日常缓存，便于 `git add -f` 入库）
    rewrite_frozen_path: str = "data/frozen/rewrite_frozen.json"

    # 数据库（在 .env 中配置 DATABASE_URL，勿在代码里写明文密码）
    database_url: str = ""

    # 数据路径
    medical_data_dir: str = "../Chinese-medical-dialogue-data-master/Data_数据"
    cleaned_data_dir: str = "data/by_department"
    # 离线评估结果目录（展示大屏读最新一次评估的指标）
    eval_dir: str = "data/eval"

    # 检索权重（eval_rag.py tune 可搜出最优值，写进 .env 或 tuned 文件生效）
    bm25_weight: float = 0.3
    vector_weight: float = 0.7
    reranker_top_k: int = 5

    # ---- 多轮会话分流（方案 B）----
    # 把「好的/谢谢/嗯」这类会话语义短路到对话路径，不做检索
    intent_gate_enabled: bool = True
    # 判为会话语的最大长度（去掉标点后）；超过就按医学问题走
    intent_chat_max_chars: int = 8

    # ---- 相关性闸门（方案 C）----
    # rerank 最高分低于此阈值 → 判定库中无相关内容，清空 context
    relevance_gate_enabled: bool = True
    # 起点值。rerank_score 是 normalize=True 后的 sigmoid 输出，落在 (0,1)；
    # 需按真实分数分布标定（见「多轮对话连贯性修复方案.md」Step 8d）
    rerank_score_threshold: float = 0.30

    # ---- 对话行为分流（阶段一）----
    # 总开关。关掉后 dialogue_act 恒按 "new_question" 处理 ——
    # 等价于本阶段之前的行为（注意改写 Prompt 已改，需一并回滚 Prompt 才完全复原）。
    dialogue_act_enabled: bool = True
    # followup（指代追问）是否改用 enhanced_query 打分。
    # 这是本阶段**唯一改变检索行为**的开关，单独留出便于 A/B：
    #   开 → 「它有什么副作用」这类被补全后才与文档可比（修 F 案例 0.355 的漏答）；
    #   关 → 全部用原始 question 打分（退回更保守的行为）。
    followup_score_with_enhanced: bool = True
    # 自足问题（new_question）是否跳过改写、直接用原句检索（阶段一 §13.9 的"接线"）。
    # 实测改写对自足问题是**净负收益**：无改写 hit_rate 0.92 / mrr 0.9067，
    # 有改写 0.70~0.82；12 题有差异、0 题反向（符号检验 p≈0.016）。
    #   开 → new_question 用原句检索，对齐冻结基线口径；followup 仍用改写补全指代。
    #   关 → 退回"改写一律生效"（改写的 LLM 调用照做，只是产物被丢弃，零额外延迟）。
    # 注意：dialogue_act_enabled=False 时本闸不生效（act 是恒定值而非真实分类，
    # 不能据此断言"这题自足"），保持本阶段之前的"改写一律生效"行为。
    rewrite_gate_on_act: bool = True

    # ---- 指代消解（阶段三）----
    # 焦点实体总开关。关掉 = 不消费 Prompt 的焦点行、不做改写退化兜底，
    # 行为退回本阶段之前（注意 Prompt 已改成三行，需一并回滚 Prompt 才完全复原）。
    # ⚠️ 依赖 dialogue_act_enabled：焦点要靠 act 判"更新还是保持"、判"是不是指代追问"，
    #    阶段一关掉时本开关自动失效（见 graph.node_rewrite 的 focus_ok）。
    focus_entity_enabled: bool = True

    # ---- 忠实性校验（阶段四）----
    # L1：引用编号可校验（确定性，零 LLM）。剥除越界编号并推 correction 事件。
    groundedness_citation_check: bool = True
    # L2：逐句忠实性核查（+1 次 LLM 调用 / 每次医学回答）。默认关——
    #     先让 L1 跑一段、看 trace 里 unsupported 的分布，确认误报率可接受再开。
    groundedness_llm_enabled: bool = False

    # ---- 检查点与人工澄清（阶段二 / 咨询稿 A3）----
    # LangGraph checkpointer 总开关。关掉 = 图不带 checkpointer 编译、
    # 路由永不进 human_review，行为与本阶段之前完全一致。
    graph_checkpointer_enabled: bool = True
    graph_checkpointer_db_path: str = "data/cache/graph_checkpoints.sqlite"
    # 证据不足时是否中断追问用户（依赖 graph_checkpointer_enabled）
    human_review_enabled: bool = True
    # 单轮最多追问次数（防「追问→还是没有→再追问」死循环）
    human_review_max_rounds: int = 1

    # ---- 节点级可靠性与预算（T62-⑧）----
    # 为什么把重试从 llm.py 的 client 层上移到图节点层：client 只知道"这次 HTTP
    # 调用失败"，不知道"这个节点在图里干什么"，于是它只能一律重试、也一律把异常
    # 抛给调用方 —— 图没有接管的机会，provider 抽一下整轮就是空响应。上移之后
    # 每个节点有自己的预算与降级动作，**任何单点失败都不再让整轮变成空响应**。
    #
    # 两个开关的默认值刻意不同，理由不同：
    #   重试/降级**只在出错时**生效，成功路径逐字不变 ⇒ 默认开；
    #   超时是另一回事，见下。
    node_retry_enabled: bool = True
    # 单节点最多尝试几次（含首次）。只作用于**非流式**节点，见 graph._NODE_KINDS
    node_retry_max_attempts: int = 3
    node_retry_initial_interval: float = 0.5
    node_retry_backoff_factor: float = 2.0
    node_retry_max_interval: float = 8.0
    #
    # ⚠️⚠️ **节点级 `TimeoutPolicy` 在本项目不可用，不要再加回来。**
    # langgraph 1.2.11 在 `compile()` 阶段就会拒绝：
    #   ValueError: Node timeouts are only supported for async nodes because sync
    #   Python execution cannot be safely cancelled in-process. Node 'rewrite' is sync.
    # 本项目所有节点都是 `def`（sync）⇒ 只要挂上 TimeoutPolicy，`build_graph()`
    # 直接抛错、**后端起不来**（2026-09-28 实测踩到）。这不是运行时降级，是
    # 最坏的一类失败：一个"看起来更安全"的开关会让服务整个消失。
    # 想压单轮墙钟只有两条路：① 把节点改成 async（大工程）；② 调**客户端级**
    # 超时 `llm_timeout_sec`（下一条，已在生效）。当前走 ②。
    #
    # 单次 LLM 请求的墙钟上限（秒），透传给 ChatOpenAI 的 timeout。
    # 它是**当前唯一真正生效的超时**：客户端的 httpx 超时能真的掐断 socket。
    # 默认 180s；配合 `node_retry_max_attempts`，一次节点最多 (180 × max_retries)
    # × max_attempts 的墙钟 —— 要压预算就同时降这两个值。
    llm_timeout_sec: float = 180.0

    # SSE 流的**空闲**看门狗（秒）：多久没往客户端推下一条事件就收尾。
    # <=0 = 不设。为什么需要它：`llm_timeout_sec` 对流式响应**不是墙钟上限** ——
    # httpx 的 read 超时会被持续吐字/心跳不断重置，实测「好痛」这种 4 字问题
    # 单轮能跑 30+ 分钟（`sse_starlette` 自己每 15s 发 `: ping`，连接始终是活的）。
    # 判据刻意用 idle 而非"整轮墙钟"：慢但在吐字的流不该被杀，要掐的是"卡住不动"。
    # 前端 `Chat.tsx` 有一层 105s 的对称兜底（阈值更大，让后端先收尾）。
    sse_idle_timeout_sec: float = 90.0

    # ---- 证据分级（T62-④）----
    # 为什么默认关：它给**每一次有资料的医学回答**加一次 LLM 调用，而 provider
    # 尾延迟是本项目当前最大的噪声源（实测 1.4s / 33.6s / 254.1s 同题三次）。
    # 与 groundedness_llm_enabled 同一纪律：先让确定性部分跑稳，再按需打开。
    #
    # 打开后它解决 README「已知限制」第一条：资料与问题**沾边但答不了**时
    # top_score 依然很高（实测 0.576 / 0.914 / 0.647，与真·可回答的 0.658~1.000
    # 完全重叠），分数切不开 ⇒ 改由 LLM 判"这批资料够不够"，不够就把这轮推进
    # 无资料路径。2026-09-28 实测：靶样本四条全部被正确翻转。
    #
    # ⚠️ 原设计里还有一条"换检索词回边再检一次"，**已实测删除**（拿不到新词，
    # 纯浪费一轮检索 + 一次 LLM，理由见 graph.node_grade）。所以这里**没有**
    # `evidence_grade_max_retries` —— 不要再把它加回来，除非先改分级 Prompt。
    evidence_grade_enabled: bool = False
    # 分级判定用零温：这是分类任务，不是生成任务。与 rewrite_temperature 同理。
    evidence_grade_temperature: float = 0.0

    # ---- 确定性医学计算工具（T62-⑤）----
    # 打开后在检索之后插一个 `tool` 节点：由一次 LLM 判断"这题要不要算"，
    # 要算则用**纯函数**算（CKD-EPI eGFR + 分期、BMI + 中国标准分类、儿童按体重给药）。
    #
    # 为什么值得单独一个节点：这类问题让模型"心算"经常错（指数项、单位换算），
    # 检索也检索不出"你这个数值对应的答案"。确定性的事交给代码，是工具节点的本意。
    #
    # 为什么默认关：同样是每轮 +1 次 LLM 调用（用于判定与抽参），与 ④ 同一纪律。
    # 任何失败/不需要都只是"没算"，不影响本轮问答（fail-open，见 medical_tools）。
    tool_node_enabled: bool = False

    # ---- 跨会话长期记忆（T62-⑦）----
    # 打开后在检索之后读一次长期记忆（慢病史/过敏史/长期用药）、在收尾时抽一轮写回。
    # 存储用 langgraph 的 SqliteStore（与 checkpointer 同源，零新依赖）。
    # 为什么默认关：写入侧要 +1 次 LLM 调用（抽取长期事实），与 ④⑤ 同一纪律。
    # 读取侧是本地 sqlite 微秒级读，不产生调用。
    #
    # ⚠️ 与 focus_entity 的分工：focus_entity 是 thread 内的（换会话就没了），
    # 本项是跨会话的。两者生命周期不同，**不要合并**。
    long_term_memory_enabled: bool = False
    long_term_memory_db_path: str = "data/graph_memory.sqlite"

    # tune 输出最优权重的落盘路径（后端自动叠加到默认值）
    tuned_weights_path: str = "data/tuned_weights.json"
    
    # 服务
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    frontend_url: str = "http://localhost:5173"

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    @property
    def data_dir_resolved(self) -> Path:
        """返回数据集绝对路径（相对 backend 目录解析）"""
        return Path(__file__).resolve().parent.parent / self.medical_data_dir

    @property
    def cleaned_data_dir_resolved(self) -> Path:
        """返回清洗后数据的绝对路径"""
        return Path(__file__).resolve().parent.parent / self.cleaned_data_dir

    @property
    def eval_dir_resolved(self) -> Path:
        """返回离线评估结果目录的绝对路径"""
        return Path(__file__).resolve().parent.parent / self.eval_dir

    @property
    def milvus_uri(self) -> str:
        return f"http://{self.milvus_host}:{self.milvus_port}"

    @property
    def upload_dir_resolved(self) -> Path:
        """返回上传目录绝对路径"""
        return Path(__file__).resolve().parent.parent / self.upload_dir

    @property
    def rewrite_cache_path_resolved(self) -> Path:
        """返回改写缓存文件绝对路径（相对 backend 解析，目录不存在时由缓存自行创建）"""
        return Path(__file__).resolve().parent.parent / self.rewrite_cache_path

    @property
    def rewrite_frozen_path_resolved(self) -> Path:
        """返回冻结改写集文件绝对路径（与日常缓存分开，便于单独入库）"""
        return Path(__file__).resolve().parent.parent / self.rewrite_frozen_path

    @property
    def reranker_model_path(self) -> str:
        """Reranker 模型的**本地快照绝对路径**；解析不到时回退为原始模型 id"""
        return resolve_hf_snapshot(self.reranker_model)


def resolve_hf_snapshot(model_id: str) -> str:
    """把 HF 模型 id 解析为本地快照的绝对路径。

    为什么需要它：并非所有库都会把 `local_files_only` 这类 kwargs 下传给
    `from_pretrained`。最典型的是 FlagEmbedding 的 `FlagReranker`——它构造
    tokenizer/model 时只传 `trust_remote_code` / `cache_dir`，其余 kwargs 全部
    丢弃，于是进程照样向 huggingface.co 发 HEAD 校验版本，国内直连必被
    SSL 阻断。改传本地绝对目录后，`from_pretrained` 直接走 `os.path.isdir`
    分支，零网络请求。

    解析顺序：`refs/main` 指向的 revision → `snapshots/` 下最新目录 → 原样返回
    model_id（调用方需自行判断是否降级告警）。
    """
    if os.environ.get("HF_HUB_CACHE"):
        cache_root = Path(os.environ["HF_HUB_CACHE"])
    elif os.environ.get("HF_HOME"):
        cache_root = Path(os.environ["HF_HOME"]) / "hub"
    else:
        cache_root = Path.home() / ".cache" / "huggingface" / "hub"

    repo_dir = cache_root / f"models--{model_id.replace('/', '--')}"
    snapshots_dir = repo_dir / "snapshots"

    ref_file = repo_dir / "refs" / "main"
    if ref_file.is_file():
        candidate = snapshots_dir / ref_file.read_text(encoding="utf-8").strip()
        if candidate.is_dir():
            return str(candidate)

    if snapshots_dir.is_dir():
        candidates = sorted(p for p in snapshots_dir.iterdir() if p.is_dir())
        if candidates:
            return str(candidates[-1])

    return model_id


settings = Settings()
