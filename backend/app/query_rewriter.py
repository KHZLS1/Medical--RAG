"""查询改写模块

在检索前对用户 query 做增强，提升召回率。
用户提问往往口语化，先用 LLM 改写成关键词增强版再去向量检索，
让 bge embedding 能更精准命中相关问答。

策略：
  - rewrite_query: 单次改写（默认），口语化 → 关键词增强版
  - multi_query:   多路改写（可选），生成 N 个变体用于多路召回

集成位置见 rag_chain.retrieve()：
  检索用改写后的 query，Reranker 与 Prompt 用原始 question。

⚠️ 输出契约（本模块最重要的一条约定）
------------------------------------------------
LLM 的自由文本**不能**直接当检索词用。曾经出过事故：LLM 返回
「（用户过于简短，无法判断具体意图）」这类说明文字，被一路送到 Milvus 当查询串。
所以改写的对外入口是 `rewrite_for_retrieval()`，它返回 `RewriteResult`：

  - `query` **一定**是可直接检索的字符串（契约保证，退化时回退原问题）；
  - 退化时 `degraded=True` + `reason` 说明原因，供 trace 观测。

收敛逻辑全部集中在纯函数 `_sanitize_rewrite()`（不碰 LLM，可离线单测）。

⚠️ 可复现性（本模块的第二条约定）
------------------------------------------------
改写是**评估链路的唯一随机源**：`hit_rate` / `mrr` 只依赖检索集，检索集只由
`enhanced_query` 决定。所以这里做两件事，缺一不可：
  1. 改写走**零温**（`settings.rewrite_temperature`，默认 0.0）；
  2. 结果落**磁盘缓存**（`app/rewrite_cache.py`）——跨进程复用，
     重跑同一测试集得到逐字相同的 query，指标才可比。
实测依据：同一份代码连跑两次，`hit_rate` 会差 4 题（0.08），单次评估分不出小改动。
"""
import hashlib
import re
from dataclasses import dataclass
from functools import lru_cache

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .config import settings
from .llm import get_llm
from .rewrite_cache import RewriteCache


# ============================================================================
# 改写输出契约
# ============================================================================
@dataclass(frozen=True)
class RewriteResult:
    """改写节点的输出契约。

    query   —— 一定是可直接用于检索的字符串（契约保证）；
    degraded —— 改写失败/无效，query 已回退为原问题；
    reason  —— 退化原因（仅观测用，改变不了行为）；
    transient —— 失败是**暂时性**的（LLM/网络异常，而非 LLM 给出了不合格输出）。
                 用来决定"要不要写进磁盘缓存"：暂时的失败绝不能固化，
                 否则一次网络抖动会让这道题以后永远用原句检索，且查不出来。
    dialogue_act —— 这一轮的对话行为（阶段一）：new_question / followup / ack / chitchat。
                 由**同一次改写调用顺带产出**（改写本来就看得见 history，零额外延迟）。
                 解析不到标签时回退 "new_question" —— 最安全：走完整检索。
    focus_entity —— 当前讨论的医学实体短语（阶段三）："高血压 党参" / "小儿发热"。
                 与 query 的区别：query 是**这一轮**的检索词，焦点是**跨轮**的状态。
                 二者刻意分开 —— query 在 new_question 时会被闸门丢弃，而焦点必须留下。
                 焦点与 query 的退化状态**解耦**：改写退化时 query 不可用，焦点照样提取。
    """
    query: str
    degraded: bool = False
    reason: str = ""
    transient: bool = False
    dialogue_act: str = "new_question"
    focus_entity: str = ""


# 元话语词表：LLM 在"改不动"时会转而解释自己改不动，这些词就是解释的标志。
# 命中即判退化。注意这里**不收**裸的「建议」——它在正常医学查询里太常见
# （"医生建议我吃…"），而漏判的代价只是多一次低质量检索（后面还有证据闸门兜着），
# 误判的代价却是把好查询换成原句。宁可漏，不可误杀。
_META_MARKERS = (
    "无法判断", "无法确定", "无法改写", "无法补全", "无法根据",
    "过于简短", "信息不足", "不足以", "缺少足够", "没有足够",
    "请提供更多", "请补充", "抱歉", "作为AI", "作为一个AI",
)

# 剥「改写后的查询：」这类前缀。**必须带冒号**才剥，否则会把以"查询"开头的
# 正常查询串吃掉。冒号后的 \s* 会把其后的换行一并吃掉。
_PREFIX_RE = re.compile(
    r"^\s*[-*#>\s]*"
    r"(?:改写后(?:的)?(?:查询|问题)|改写结果|检索(?:查询|词)|优化后(?:的)?查询|查询|Query)"
    r"\s*[:：]\s*",
    re.IGNORECASE,
)

# 剥首尾成对包裹符（引号/书名号/括号），LLM 很爱加。
_WRAP_RE = re.compile(r"""^[「『“”‘’"'《【（(\[]+|[」』“”‘’"'》】）)\]]+$""")

# 剥 markdown 代码围栏（含可选语言标注）
_CODE_FENCE_RE = re.compile(r"```[\w+-]*\n?")


def _sanitize_rewrite(raw: str, original: str, max_chars: int | None = None) -> RewriteResult:
    """把 LLM 的自由文本收敛成契约（纯函数，零 LLM 依赖）。

    清洗顺序（顺序有讲究，别调换）：
      1. 剥代码围栏
      2. 剥「改写后的查询：」前缀 —— **先于**取第一行。反例：
         `改写后的查询：\\n高血压 党参` 若先取行会得到空串而误判退化。
      3. 只取第一行 —— 多行输出基本是解释而非查询
      4. 剥成对引号/括号
    再判三条退化规则：清洗后为空 / 超长 / 命中元话语词表。

    已知不覆盖的情况：LLM 输出一句通顺的完整句子（如"用户的问题是高血压能否吃党参"）
    不会被拦下。它含关键词，检索质量只是略差而非崩塌，交由下游证据闸门判定即可，
    不值得再堆启发式。
    """
    limit = settings.rewrite_max_chars if max_chars is None else max_chars

    text = (raw or "").strip()
    if not text:
        return RewriteResult(original, True, "LLM 返回空")

    text = _CODE_FENCE_RE.sub("", text)
    text = _PREFIX_RE.sub("", text).strip()
    text = text.split("\n", 1)[0].strip()
    text = _WRAP_RE.sub("", text).strip()

    if not text:
        return RewriteResult(original, True, "清洗后为空")
    if len(text) > limit:
        return RewriteResult(original, True, f"长度 {len(text)} 超上限 {limit}")
    hit = next((m for m in _META_MARKERS if m in text), "")
    if hit:
        return RewriteResult(original, True, f"元话语「{hit}」")
    return RewriteResult(text, False, "")


# ============================================================================
# 对话行为标签（阶段一）—— 与改写结果由同一次 LLM 调用产出
# ============================================================================
# 允许的取值。组合成策略表见 app/dialogue_policy.py。
DIALOGUE_ACTS = ("new_question", "followup", "ack", "chitchat")

# 模型可能输出中文标签，统一映射到英文枚举值；映射不到 → 回退 new_question
_ACT_LABELS = {
    "new_question": "new_question", "医学提问": "new_question", "新问题": "new_question",
    "followup": "followup", "指代追问": "followup", "追问": "followup",
    "ack": "ack", "确认附和": "ack", "确认": "ack",
    "chitchat": "chitchat", "寒暄闲聊": "chitchat", "闲聊": "chitchat",
}

# 只匹配「整行就是一个标签行」：带 $、不跨行 ——
# 正常查询串里出现「行为」二字不会被误剥（末段要求 (\S+) 后直接到行尾，
# 所以"行为：攻击性行为 儿童"这类真查询不会命中）。
#
# ⚠️ 标签词两侧与取值两侧都允许 `*`：模型极爱输出 `**行为**：followup`。
# 若不容忍加粗，这行就匹配不上 → 落到 _sanitize_rewrite 被当成第一行检索词，
# 于是 `**行为**：followup` 被送进 Milvus —— 正是本模块要防的那类事故。
_ACT_LINE_RE = re.compile(
    r"^\s*[-*#>\s]*\**\s*(?:行为|类型|对话行为|act)\**\s*[:：]\s*\**\s*(\S+?)\s*\**\s*$",
    re.IGNORECASE,
)

# 查询标签行：与 _ACT_LINE_RE 同形，但取值允许含空格（查询本身有空格）。
# 命中即**直接用它的取值当 query**，不再依赖 _sanitize_rewrite 里的 _PREFIX_RE ——
# 后者不认加粗（`**查询**：xxx`）也不认「问题」这类标签，
# 会让 `**查询**：高血压` 整串当检索词（前面加粗的行为标签会因 _LABEL_RESIDUE_RE
# 被降级回原问题，而加粗的查询标签只会静默降低检索质量，比前者更隐蔽）。
_QUERY_LINE_RE = re.compile(
    r"^\s*[-*#>\s]*\**\s*"
    r"(?:查询|问题|检索(?:词|查询)|改写后(?:的)?(?:查询|问题)|改写结果|Query)"
    r"\**\s*[:：]\s*\**\s*(.+?)\s*\**\s*$",
    re.IGNORECASE,
)

# 兜底网：万一仍出现没预料到的标签格式（例：`对话行为 - followup`、`【对话行为】 followup`），
# 剥不掉的标签行依然会被当检索词。这里对**收敛后**的结果再判一次：
# 整串形如「<标签词><修饰>[:：]?<已知取值>」即认为解析失败 → 回退原问题（宁可少赚，不可投毒）。
_LABEL_RESIDUE_RE = re.compile(
    r"^\W*(?:行为|类型|对话行为|act)\W*[:：]?\s*(\S+?)\s*\W*$",
    re.IGNORECASE,
)

# 「焦点：」标签行（阶段三）。与行为/查询标签行同形，取值允许含空格（实体短语有空格）。
_FOCUS_LINE_RE = re.compile(
    r"^\s*[-*#>\s]*\**\s*(?:焦点|当前实体|讨论实体|实体|focus)\**\s*[:：]\s*\**\s*(.+?)\s*\**\s*$",
    re.IGNORECASE,
)

# 焦点取值的「空」写法。必须**精确匹配**（不能子串匹配）：
# "无" 是合法空值，但 "无痛" 是实体 —— 子串匹配会把后者误判成空。
_FOCUS_EMPTY = frozenset({"无", "空", "none", "n/a", "na", "null", "-", "—", "（无）", "(无)"})

# 焦点长度上限：实体短语不该超过 30 字。超长说明模型在解释而非给实体 → 宁缺勿滥。
_FOCUS_MAX_CHARS = 30


def _clean_focus(raw: str) -> str:
    """把「焦点」标签的取值收敛成实体短语（纯函数，零 LLM）。

    与 `_sanitize_rewrite` 的两处刻意差异：
      1. **不做元话语判定**——"无"这类空值本身就是合法输出，不是"模型在解释";
      2. 超长直接判空而非回退原问题——焦点缺失是安全状态（阶段三的兜底会退回
         下一层），而一个被污染的超长"焦点"会被拼进检索词，比没有更糟。
    """
    text = _CODE_FENCE_RE.sub("", (raw or "").strip())
    text = text.split("\n", 1)[0].strip()
    text = _WRAP_RE.sub("", text).strip()
    if not text or text.lower() in _FOCUS_EMPTY:
        return ""
    if len(text) > _FOCUS_MAX_CHARS:
        return ""
    return text


def _parse_rewrite_output(raw: str, original: str) -> RewriteResult:
    """解析 Prompt 约定的输出：`行为：<act>` / `焦点：<entity>` / `查询：<query>` → 契约对象。

    逐行扫描，把**标签行吃掉**，只把剩下的内容交给 `_sanitize_rewrite` 收敛。

    ⚠️ 为什么必须先剥标签再收敛：`_sanitize_rewrite` 的第 3 步是"只取第一行"。
    标签行若留在最前面，就会被当成检索词送进 Milvus —— 正是这条链路最初出事故
    的那个形态（LLM 的说明文字被当查询串）。焦点行同理，不消费它就会变成
    `"焦点：高血压 党参"` 整串进 Milvus。

    四处降级都不破坏 query 契约（保证可直接检索）：
      标签行缺失（旧格式输出）→ act 回退 new_question，其余照常收敛；
      标签值非法              → 同上；
      标签不在第一行          → 逐行扫描仍能识别，query 不受污染；
      标签格式没预料到        → _LABEL_RESIDUE_RE 兜底，直接回退原问题。
    向后兼容：没有标签行时，走的分支与加标签之前**逐字相同**（kept 拼接 → _sanitize_rewrite）。
    """
    act = "new_question"
    seen_tag = False
    query_from_label: str | None = None
    focus_raw: str | None = None
    kept: list[str] = []

    for line in (raw or "").split("\n"):
        m = _ACT_LINE_RE.match(line)
        if m:
            # 只认第一条行为标签行；重复的标签行一律丢弃（它绝不会是查询内容）
            if not seen_tag:
                seen_tag = True
                act = _ACT_LABELS.get(m.group(1).strip().lower(), "new_question")
            continue
        m = _FOCUS_LINE_RE.match(line)
        if m:
            # 焦点行同样必须被消费，否则整串会落进 kept 被当检索词（见 docstring）
            if focus_raw is None:
                focus_raw = m.group(1)
            continue
        m = _QUERY_LINE_RE.match(line)
        if m:
            if query_from_label is None:
                query_from_label = m.group(1).strip()
            continue
        kept.append(line)

    # 有「查询：」标签行 → 只认它（这正是 Prompt 约定的格式）；
    # 没有（旧格式）→ 其余内容整体交给 _sanitize_rewrite，行为与本阶段之前一致。
    candidate = query_from_label if query_from_label else "\n".join(kept)
    base = _sanitize_rewrite(candidate, original)

    # 兜底网：收敛后仍是"标签行"形态 → 说明上面没剥干净，这串东西绝不能当检索词。
    if not base.degraded:
        m = _LABEL_RESIDUE_RE.match(base.query)
        if m and m.group(1).strip().lower() in _ACT_LABELS:
            return RewriteResult(original, True, "标签行未被剥离", base.transient, act,
                                 _clean_focus(focus_raw or ""))

    return RewriteResult(base.query, base.degraded, base.reason, base.transient, act,
                         _clean_focus(focus_raw or ""))


def _invoke_rewrite(prompt_text: str, payload: dict, original: str) -> RewriteResult:
    """跑一次改写 LLM 调用并收敛成契约。

    异常在这里被吃掉、转成 degraded 的 RewriteResult —— 因为它被 lru_cache 包着，
    "失败"也要进缓存，否则同一个问题每次重试都要白等一次 1~2s 的失败调用。

    温度取 `settings.rewrite_temperature`（默认 0.0）：改写是关键词抽取，
    不需要创造性，而 0.3 会让同一问题每次产出不同的关键词、评估指标随之漂移。
    """
    try:
        llm = get_llm(temperature=settings.rewrite_temperature)
        chain = ChatPromptTemplate.from_template(prompt_text) | llm | StrOutputParser()
        raw = chain.invoke(payload)
    except Exception as e:
        print(f"[查询改写] 失败，回退原始问题: {e}")
        # transient=True：这是网络/服务端异常，不是"LLM 输出不合格"。
        # 这种结果**不进磁盘缓存**，下次调用重新试。
        return RewriteResult(original, True, f"改写异常：{type(e).__name__}", transient=True)
    return _parse_rewrite_output(str(raw), original)


# ============================================================================
# 改写磁盘缓存（跨进程 / 跨运行复用，评估可比性的前提）
# ============================================================================
_cache: RewriteCache | None = None
_cache_ready = False


def _prompt_fingerprint() -> str:
    """改写 Prompt + 模型 + 温度 的指纹。任一变化 → 缓存整体作废。

    在**调用时**才读模块级 Prompt 常量（此处定义位置早于它们），
    Python 的全局解析是运行时的，所以顺序没问题。
    """
    material = "\x00".join([
        REWRITE_PROMPT,
        CONTEXT_REWRITE_PROMPT,
        settings.llm_model,
        str(settings.rewrite_temperature),
    ])
    return hashlib.sha1(material.encode("utf-8")).hexdigest()[:16]


def _get_cache() -> RewriteCache | None:
    """惰性单例。未启用时返回 None，调用方按"无缓存"处理。"""
    global _cache, _cache_ready
    if _cache_ready:
        return _cache
    _cache_ready = True
    if settings.rewrite_cache_enabled:
        _cache = RewriteCache(
            settings.rewrite_cache_path_resolved,
            fingerprint=_prompt_fingerprint(),
            max_entries=settings.rewrite_cache_max_entries,
        )
    return _cache


def rewrite_cache_stats() -> dict:
    """缓存命中统计（给评估脚本报告用，判断这次重跑是否复用了旧改写）"""
    if not settings.rewrite_cache_enabled:
        return {"enabled": False}
    cache = _get_cache()
    return {"enabled": True, **(cache.stats if cache else {})}


def _rewrite_cached(kind: str, history_key: str, question: str,
                    prompt_text: str, payload: dict) -> RewriteResult:
    """两处改写的公共外壳：查磁盘缓存 → 未命中才调 LLM → 落盘。

    磁盘缓存的意义和 lru_cache 不同：后者只活在单个进程内，评估脚本每次重跑都是
    全新的进程、全新的一次 LLM 调用，于是 enhanced_query 每次都不同、指标不可比。
    """
    cache = _get_cache()
    if cache is not None:
        hit = cache.get(kind, history_key, question)
        if hit is not None:
            return RewriteResult(
                hit["query"], bool(hit.get("degraded")), hit.get("reason", ""),
                # 旧缓存条目没有 act 字段 → 回退 new_question（最安全：走完整检索）
                dialogue_act=hit.get("act", "new_question"),
                # 旧缓存条目没有 focus 键 → 空串（阶段三兜底会退回下一层，安全）
                focus_entity=hit.get("focus", ""),
            )

    result = _invoke_rewrite(prompt_text, payload, question)
    # 只固化"有效结果"和"契约退化"（LLM 确实给出了不合格输出，属确定性事实）。
    # transient（异常/超时）不写盘 —— 否则一次抖动会永久固化成"这道题不改写"。
    if cache is not None and not result.transient:
        cache.put(kind, history_key, question,
                  result.query, result.degraded, result.reason, result.dialogue_act,
                  result.focus_entity)
    return result

REWRITE_PROMPT = """你是一个医学搜索查询改写助手。请先判断这轮发言的「对话行为」，再改写查询。

【对话行为】必选其一，原样输出以下 4 个英文词之一：
- new_question：一个自足的医学提问（新话题，或话题切换）
- followup：指代追问，必须结合上文才成立（含"它/这个病/那/还要/副作用"等指代）
- ack：确认、附和或收尾（"好的/嗯/不了/算了/用不着/我再想想"）
- chitchat：寒暄闲聊（"你好/谢谢/再见/你是谁"）

【症状主诉不是闲聊】这是最容易判错的一类，务必分清：
- "好困/好疼/好难受/好胀/有点晕/睡不着"这类**纯症状主诉**，哪怕只有两三个字、
  没有问句形式，也一律**按医学问题处理**（无上文时算 new_question；有上文且明显
  承接上文时算 followup），**绝不能判成 ack 或 chitchat** —— 用户是在报告症状，
  要检索作答，判成闲聊就等于漏答。
- 「好」字打头的**确认词**（好 / 好的 / 好吧 / 好嘞）才算 ack；
  「好」后面跟着症状词的（好困 / 好疼 / 好恶心）**不是** ack。
- chitchat 仅限：问候、致谢、道别、询问你是谁 / 你能做什么。
  **症状或身体不适的抱怨一律不算 chitchat。**

【改写要求】
1. 提取核心医学关键词（疾病、症状、药物、检查等）
2. 补充相关医学术语与同义词
3. 去除"能不能""可以吗"等无意义口语词汇
4. 保持原意不变，不引入用户未提及的新病症
5. 即使问题很短、不像医学问题，也必须给出查询串本身；**严禁**输出
   "无法判断""信息不足""过于简短"之类的说明——实在改不动就原样输出用户问题。
6. 「焦点」是**跨轮状态**，不是这一轮的检索词：它回答"我们正在聊哪个病/哪个药"。
   话题切换时要换成新实体；纯寒暄或没有明确实体时写「无」。
   示例：上一轮"高血压患者能吃党参吗" → 焦点"高血压 党参"；
        本轮"它有什么副作用" → 焦点"党参 副作用"，查询"党参 副作用 禁忌"。

【输出格式】严格三行，不要任何其它内容、不要编号、不要加引号：
行为：<new_question|followup|ack|chitchat>
焦点：<当前正在讨论的核心疾病/症状/药物实体，短语，不超过20字；没有明确实体就写「无」>
查询：<改写后的查询>

用户问题：{question}
"""

@lru_cache(maxsize=256)
def _single_rewrite(question: str) -> RewriteResult:
    """单轮改写（带缓存）→ 契约对象。无历史时使用。

    两级缓存：进程内 lru_cache（省重复调用）+ 磁盘缓存（跨进程可复现）。
    """
    return _rewrite_cached(
        "single", "", question, REWRITE_PROMPT, {"question": question}
    )


def rewrite_query(question: str) -> str:
    """单次改写：口语化问题 → 关键词增强版（兼容入口，返回裸字符串）

        示例：
          "高血压患者能吃党参吗" → "高血压 党参 用药禁忌 药物相互作用 安全性"
        输出已过契约收敛：返回值**一定**能直接拿去检索，退化时即原问题。
        eval_rag.py 依赖本函数的 str 签名，不要改成返回 RewriteResult。
        """
    if not settings.query_rewrite_enabled:
        return question
    return _single_rewrite(question).query

MULTI_QUERY_PROMPT = """你是医学搜索助手。请为以下问题生成 {n} 个不同角度的检索查询变体，用于多路召回。
要求：
1. 每行一个变体，不要编号、不要解释
2. 角度可包括：疾病机制、用药安全、检查指标、日常注意事项等
3. 保持与原问题相关，不引入无关病症

用户问题：{question}
{n} 个检索变体："""

def multi_query(question: str, n: int = 3) -> list[str]:
    """多路改写：生成 n 个变体 query（含原始 query）用于多路召回

        示例：
          原始："高血压患者能吃党参吗"
          变体1："党参 高血压 禁忌 相互作用"
          变体2："高血压中药调理 党参安全性"
          变体3："高血压患者 人参 党参 用药注意"
        """
    try:
        llm = get_llm()
        prompt = ChatPromptTemplate.from_template(MULTI_QUERY_PROMPT)
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({"question": question, "n": n})
        queries = [q.strip() for q in result.strip().split("\n") if q.strip()]
        queries.insert(0, question)
        return queries[:n + 1]
    except Exception as e:
        print(f"[多路改写] 失败，回退原始问题: {e}")
        return [question]

CONTEXT_REWRITE_PROMPT = """你是一个医学对话上下文理解助手。
请先判断用户当前这句话的「对话行为」，再结合对话历史把它改写成一个完整、独立的检索查询。

【对话行为】必选其一，原样输出以下 4 个英文词之一：
- new_question：一个自足的医学提问（新话题，或话题切换）
- followup：指代追问，必须结合上文才成立（含"它/这个病/那/还要/副作用"等指代）
- ack：确认、附和或收尾（"好的/嗯/不了/算了/用不着/我再想想"）
- chitchat：寒暄闲聊（"你好/谢谢/再见/你是谁"）

【症状主诉不是闲聊】这是最容易判错的一类，务必分清：
- "好困/好疼/好难受/好胀/有点晕/睡不着"这类**纯症状主诉**，哪怕只有两三个字、
  没有问句形式，也一律**按医学问题处理**（无上文时算 new_question；有上文且明显
  承接上文时算 followup），**绝不能判成 ack 或 chitchat** —— 用户是在报告症状，
  要检索作答，判成闲聊就等于漏答。
- 「好」字打头的**确认词**（好 / 好的 / 好吧 / 好嘞）才算 ack；
  「好」后面跟着症状词的（好困 / 好疼 / 好恶心）**不是** ack。
- chitchat 仅限：问候、致谢、道别、询问你是谁 / 你能做什么。
  **症状或身体不适的抱怨一律不算 chitchat。**

【改写要求】
1. 补全省略的主语、指代（如"它""这个病""上述症状"等）
2. 结合之前讨论的疾病/症状，使改写后的问题可以独立被检索
3. 保持医学专业性，提取关键术语
4. 即使当前问题很短、与上文无关或不像医学问题，也必须输出一条查询串；
   **严禁**输出"无法判断""信息不足""过于简短"之类的说明——若真的无从结合
   上下文，就原样输出当前问题，不要改写。
5. 「焦点」是**跨轮状态**，不是这一轮的检索词：它回答"我们正在聊哪个病/哪个药"。
   话题切换时要换成新实体；纯寒暄或没有明确实体时写「无」。
   示例：上一轮"高血压患者能吃党参吗" → 焦点"高血压 党参"；
        本轮"它有什么副作用" → 焦点"党参 副作用"，查询"党参 副作用 禁忌"。

【输出格式】严格三行，不要任何其它内容、不要编号、不要加引号：
行为：<new_question|followup|ack|chitchat>
焦点：<当前正在讨论的核心疾病/症状/药物实体，短语，不超过20字；没有明确实体就写「无」>
查询：<改写后的查询>

【对话历史】
{history}

【当前问题】
{question}
"""

def format_history(history: list[dict], max_turns: int = 6) -> str:
    """将历史消息格式化为文本，只取最近 max_turns 条"""
    # 取最近 N 条（用户+助手算2条，所以6条约3轮）
    recent = history[-max_turns:] if len(history) > max_turns else history
    lines = []
    for msg in recent:
        role_label = "用户" if msg["role"] == "user" else "助手"
        lines.append(f"{role_label}: {msg['content']}")
    return "\n".join(lines) if lines else "（无历史对话）"

@lru_cache(maxsize=256)
def _context_rewrite(history_key: str, question: str) -> RewriteResult:
    """带缓存的上下文改写：lru_cache 要求键可哈希，
    故把 history 先格式化成字符串再作为缓存键（与 Prompt 入参一致）。"""
    return _rewrite_cached(
        "context",
        history_key,
        question,
        CONTEXT_REWRITE_PROMPT,
        {"history": history_key, "question": question},
    )


def rewrite_for_retrieval(question: str, history: list[dict] | None = None) -> RewriteResult:
    """⚠️ 改写节点的**唯一入口**（graph.node_rewrite 用它）→ 返回契约对象。

    退化时的行为（D1 选 A）：**老实回退原问题**，不拼历史主题。
    「不了」就让它以「不了」去检索 —— 检索不出东西是正确答案，
    由下游的证据闸门判成 evidence_state=none 走兜底节点。
    真正的指代追问改写失败时召回会变差，但**不会错答** —— 这一缺口由阶段三的
    focus_entity 兜底补上（见 graph.node_rewrite 的 focus_fallback 分支）。

    注意 focus_entity 与本函数的退化状态**解耦**：即使这里 degraded=True，
    焦点仍会被提取出来，正是为了给上面那条兜底用。
    """
    if not settings.query_rewrite_enabled:
        return RewriteResult(question)

    if not history:
        return _single_rewrite(question)

    # history 统一格式化成字符串，同时作为缓存键（可哈希）与 Prompt 入参
    return _context_rewrite(format_history(history), question)


def rewrite_query_with_context(question: str, history: list[dict] | None = None) -> str:
    """结合多轮历史改写当前问题，返回可独立检索的完整 query

    Args:
        question: 当前用户问题
        history: 历史消息列表，格式 [{"role": "user", "content": "..."}, ...]

    Returns:
        改写后的完整查询字符串（已过契约收敛）
    """
    return rewrite_for_retrieval(question, history).query


TITLE_PROMPT = """请为以下医学问答生成一个简短的对话标题（不超过15个字）。
要求：
1. 概括用户问题的核心内容（疾病、症状、药物等）
2. 简洁明了，不要加引号、不要加句号、不要换行
3. 只输出标题文本，不要任何解释

用户问题：{question}
对话标题："""


def generate_title(question: str) -> str:
    """用 LLM 生成简短对话标题，失败时回退到截取前20字"""
    try:
        llm = get_llm()
        prompt = ChatPromptTemplate.from_template(TITLE_PROMPT)
        chain = prompt | llm | StrOutputParser()
        title = chain.invoke({"question": question}).strip()
        title = title.split("\n")[0].strip()
        if len(title) > 30:
            title = title[:30]
        return title or question[:20]
    except Exception as e:
        print(f"[标题生成] 失败，回退截取: {e}")
        return question[:20] + ("..." if len(question) > 20 else "")