"""对话行为 × 证据状态 → 输出策略（唯一决策表）

为什么要有这个模块
------------------
「用户这句话该怎么答」原本散落在 graph.py 各处的 if 里：入口判要不要检索、
精排判用哪个 query 打分、证据判走哪个 Prompt。这些判断**互相牵连**却写在不同地方，
改一处很容易漏另一处（原始 bug 就是这么回归的）。

这里把它收敛成一张可读的表：一处定义、可直接对照、可单测。
本模块只描述"该怎么答"，**不负责执行** —— 执行仍在 graph 的节点里。

矩阵（`事实来源` 见 `实施计划_阶段一_对话行为与策略矩阵.md` §1.2）
----------------------------------------------------------------
  dialogue_act   | 检索 | 检索用      | rerank 打分用   | 证据 strong        | 证据 none        | 焦点（阶段三）
  ---------------|------|-------------|-----------------|--------------------|------------------|---------------
  new_question   |  是  | 原 question | 原 question     | MEDICAL（引用+声明）| INSUFFICIENT（无引用+声明）| update
  followup       |  是  | enhanced_q  | enhanced_query  | MEDICAL（引用+声明）| INSUFFICIENT（无引用+声明）| update
  ack            |  否  | —           | —               | CHAT（无引用无声明）| 同左             | keep
  chitchat       |  否  | —           | —               | CHAT（无引用无声明）| 同左             | keep

焦点列（阶段三）只有两个动作：`update`（医学类，允许覆盖焦点实体）/ `keep`
（会话语，保持原焦点）。为什么 `ack`/`chitchat` 必须是 `keep`：一句「好的」的焦点
多半是"无"，若允许覆盖就会把上一轮攒下的焦点清空，紧接着的真追问
「它有什么副作用」就没得补了 —— 这是"状态污染"最典型的一击。

关键不等式（三条必须同时成立，否则已修 bug 回归）
------------------------------------------------
1. `followup` 用改写 query 检索+打分 —— 指代被补全后才有可比性（修 F 案例 0.355 的漏答）。
2. `ack` / `chitchat` 在检索前掉头 —— 否则「不了」被改写补全成病症查询后，
   第 1 条的"改用改写打分"会让它**高分命中**「小儿发热」，反而比现状更隐蔽。
3. `new_question` 用原句检索（丢弃改写产物）—— 自足问题改写是**净负收益**：
   实测无改写 hit_rate 0.92 / mrr 0.9067，有改写 0.70~0.82；12 题有差异、
   0 题反向（符号检验 p≈0.016，见阶段一 §13.9）。
   ⚠️ 注意第 3 条与第 1 条方向相反，是刻意的不对称：改写只服务于
   "把省略/指代补全"，而不是"把问题重述一遍"。若哪天想统一，
   先看 §13.9 的数据再动手。
"""

# act -> 策略。值只描述意图，具体执行由 graph 节点完成。
# retrieve_with: 送进 Milvus 的 query 取哪一个（"question" 原句 / "enhanced_query" 改写产物）
# focus:        这一轮是否允许覆盖焦点实体（阶段三）："update" / "keep"
_RETRIEVE_POLICY: dict[str, dict] = {
    "new_question": {"retrieve": True,  "retrieve_with": "question",
                     "score_with": "question",       "answer": "medical",
                     "focus": "update"},
    "followup":     {"retrieve": True,  "retrieve_with": "enhanced_query",
                     "score_with": "enhanced_query", "answer": "medical",
                     "focus": "update"},
    "ack":          {"retrieve": False, "retrieve_with": None,
                     "score_with": None,             "answer": "chat",
                     "focus": "keep"},
    "chitchat":     {"retrieve": False, "retrieve_with": None,
                     "score_with": None,             "answer": "chat",
                     "focus": "keep"},
}

# 未知 / 解析失败的 act 一律按 new_question 处理：走完整检索。
# 漏答比多答贵 —— 分类器出错时宁可多检索一次，也不要把真问题挡在检索之外。
_DEFAULT_ACT = "new_question"

# 需要跳出检索链路的 act（子图内提前结束）
NON_RETRIEVAL_ACTS: frozenset[str] = frozenset(
    act for act, p in _RETRIEVE_POLICY.items() if not p["retrieve"]
)


def policy_for(act: str | None) -> dict:
    """取对话行为对应的策略；未知 act 回退 new_question 策略。"""
    return _RETRIEVE_POLICY.get(act or "", _RETRIEVE_POLICY[_DEFAULT_ACT])


def should_retrieve(act: str | None) -> bool:
    """这一轮要不要走检索链路。"""
    return bool(policy_for(act)["retrieve"])


def should_update_focus(act: str | None) -> bool:
    """这一轮是否允许覆盖焦点实体（阶段三 D2：仅医学类 act 更新）。

    ⚠️ 与 `should_retrieve` / `policy_for` 的"未知 act 回退 new_question"**刻意不同**：
    这里用**精确成员判定**，未知 / 缺失的 act 一律返回 False（不更新焦点）。
    理由：回退语义在"要不要检索"上是保守的（宁可多检索一次），但在"要不要覆盖
    **跨轮状态**"上恰恰是**污染源** —— 一个没认出来的 act 不该有权改写焦点。
    不更新时焦点保持原值，且 D4 的第二层兜底（`_focus_from_history`）仍在，不会漏答。
    """
    policy = _RETRIEVE_POLICY.get(act or "")
    return bool(policy) and policy.get("focus") == "update"


def all_acts() -> tuple[str, ...]:
    """已定义策略的全部 act（供脚本校验样本集里的标签拼写）。"""
    return tuple(_RETRIEVE_POLICY.keys())
