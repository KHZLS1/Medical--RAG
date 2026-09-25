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
  dialogue_act   | 检索 | rerank 打分用   | 证据 strong        | 证据 none
  ---------------|------|-----------------|--------------------|------------------
  new_question   |  是  | 原 question     | MEDICAL（引用+声明）| INSUFFICIENT（无引用+声明）
  followup       |  是  | enhanced_query  | MEDICAL（引用+声明）| INSUFFICIENT（无引用+声明）
  ack            |  否  | —               | CHAT（无引用无声明）| 同左
  chitchat       |  否  | —               | CHAT（无引用无声明）| 同左

关键不等式（两条必须同时成立，否则原始 bug 回归）
------------------------------------------------
1. `followup` 用改写 query 打分 —— 指代被补全后才有可比性（修 F 案例 0.355 的漏答）。
2. `ack` / `chitchat` 在检索前掉头 —— 否则「不了」被改写补全成病症查询后，
   第 1 条的"改用改写打分"会让它**高分命中**「小儿发热」，反而比现状更隐蔽。
"""

# act -> 策略。值只描述意图，具体执行由 graph 节点完成。
_RETRIEVE_POLICY: dict[str, dict] = {
    "new_question": {"retrieve": True,  "score_with": "question",       "answer": "medical"},
    "followup":     {"retrieve": True,  "score_with": "enhanced_query", "answer": "medical"},
    "ack":          {"retrieve": False, "score_with": None,             "answer": "chat"},
    "chitchat":     {"retrieve": False, "score_with": None,             "answer": "chat"},
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


def all_acts() -> tuple[str, ...]:
    """已定义策略的全部 act（供脚本校验样本集里的标签拼写）。"""
    return tuple(_RETRIEVE_POLICY.keys())
