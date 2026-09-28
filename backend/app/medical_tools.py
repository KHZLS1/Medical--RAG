"""确定性医学计算工具（T62-⑤ 工具节点）

为什么需要它
    医学里有一批量是**有确定公式**的 —— eGFR、BMI、按体重给药剂量。这类问题：
      · 让模型"心算"经常算错（指数项、单位换算尤其容易翻车）；
      · 去检索也检索不出"你这个具体数值对应的答案"（语料里不会有 42.6 这个数）。
    正解是**算**。这正是 Agentic RAG 里"工具节点"的用武之地：
    确定性的事交给代码，不确定性的事才交给生成。

设计纪律（与项目其它开关一致）
    1. 工具本体是**纯函数**：零 LLM、零 IO、可离线单测。这是它与"再调一次模型"
       的本质区别，也是它值得被信任的理由。
    2. **调哪个工具、参数是多少，由一次 LLM 结构化输出决定**（自然语言表述太散，
       正则抽不稳）；但这次调用只回答"要不要算、算什么、参数是多少"。
    3. **解析失败 / 参数缺失 ⇒ 不调任何工具**（fail-open）。工具层绝不能变成新的
       故障点 —— 宁可少算一次，不可拦下一轮正常问答。
    4. 结果永远带**免责声明**，且只作参考，不替代医生判断。
    5. 开关 `TOOL_NODE_ENABLED` 默认关（打开后每轮 +1 次 LLM 调用）。

⚠️ 单位约定
    肌酐入参统一走 `creatinine_to_mg_dl()` 归一，**函数内部一律 mg/dL**。
    μmol/L → mg/dL 用 88.4，这是国内化验单最常见的单位，漏换算会让 eGFR 差 ~10 倍。
"""
from __future__ import annotations

import json
import re

from .llm import get_llm

# 免责声明：所有工具结果都要带上，措辞与 no-evidence 路径保持一致。
DISCLAIMER = "以上为按标准公式的换算结果，仅供参考，不能替代医生的诊断与处方。"

_UMOL_PER_MGDL = 88.4


# --------------------------------------------------------------------------
# 纯计算层（无 LLM、无 IO —— 单测直接打这里）
# --------------------------------------------------------------------------

def creatinine_to_mg_dl(value: float, unit: str = "mg/dL") -> float:
    """把肌酐归一为 mg/dL。支持 mg/dL 与 μmol/L（含 umol/L、μmol/l 等写法）。"""
    u = (unit or "").strip().lower().replace("μ", "u").replace("µ", "u")
    if u in ("umol/l", "umol", "μmol/l"):
        return value / _UMOL_PER_MGDL
    return value


def _norm_sex(sex: str) -> str:
    s = (sex or "").strip().lower()
    if s in ("male", "m", "男", "男性"):
        return "male"
    if s in ("female", "f", "女", "女性"):
        return "female"
    return ""


def ckd_stage(egfr: float) -> str:
    """KDIGO 的 GFR 分期（G1~G5）。"""
    if egfr >= 90:
        return "G1（肾功能正常或偏高）"
    if egfr >= 60:
        return "G2（轻度下降）"
    if egfr >= 45:
        return "G3a（轻中度下降）"
    if egfr >= 30:
        return "G3b（中重度下降）"
    if egfr >= 15:
        return "G4（重度下降）"
    return "G5（肾衰竭）"


def egfr_ckd_epi(scr_mg_dl: float, age: float, sex: str) -> float:
    """CKD-EPI 2021（**无种族项**）估算肾小球滤过率，单位 mL/min/1.73m²。

    公式：142 × min(Scr/κ,1)^α × max(Scr/κ,1)^-1.200 × 0.9938^Age × (1.012 若女性)
        κ = 0.7（女）/ 0.9（男）；α = -0.241（女）/ -0.302（男）
    ⚠️ 2021 版删掉了种族系数，这是本公式与旧版最容易混淆的地方。
    """
    s = _norm_sex(sex)
    if not s:
        raise ValueError("性别必须能识别为男/女（male/female）")
    if scr_mg_dl <= 0 or age <= 0:
        raise ValueError("肌酐与年龄必须为正数")
    kappa = 0.7 if s == "female" else 0.9
    alpha = -0.241 if s == "female" else -0.302
    ratio = scr_mg_dl / kappa
    egfr = (142.0 * (min(ratio, 1.0) ** alpha) * (max(ratio, 1.0) ** -1.200)
            * (0.9938 ** age))
    if s == "female":
        egfr *= 1.012
    return egfr


def bmi(weight_kg: float, height_cm: float) -> float:
    """体重指数 BMI = 体重(kg) / 身高(m)²。"""
    if weight_kg <= 0 or height_cm <= 0:
        raise ValueError("体重与身高必须为正数")
    h = height_cm / 100.0
    return weight_kg / (h * h)


def bmi_category(value: float) -> str:
    """**中国成人**标准分类（与 WHO 国际标准不同，别混用）。

    WHO 的肥胖切点是 30，中国是 28；超重中国是 24、WHO 是 25。
    本项目面向中文语料，一律用中国标准。
    """
    if value < 18.5:
        return "偏瘦"
    if value < 24:
        return "正常"
    if value < 28:
        return "超重"
    return "肥胖"


def pediatric_acetaminophen_dose(weight_kg: float) -> dict:
    """儿童对乙酰氨基酚（扑热息痛）按体重剂量：10~15 mg/kg/次，q4~6h，24h ≤4 次。"""
    if weight_kg <= 0:
        raise ValueError("体重必须为正数")
    lo, hi = 10 * weight_kg, 15 * weight_kg
    return {
        "per_dose_mg": (round(lo, 1), round(hi, 1)),
        "interval_h": "4~6 小时一次",
        "max_per_day": "24 小时内不超过 4 次，且总量不超过 60 mg/kg/日",
    }


def pediatric_ibuprofen_dose(weight_kg: float) -> dict:
    """儿童布洛芬按体重剂量：5~10 mg/kg/次，q6~8h，24h ≤4 次；<6 月龄不推荐。"""
    if weight_kg <= 0:
        raise ValueError("体重必须为正数")
    lo, hi = 5 * weight_kg, 10 * weight_kg
    return {
        "per_dose_mg": (round(lo, 1), round(hi, 1)),
        "interval_h": "6~8 小时一次",
        "max_per_day": "24 小时内不超过 4 次，且总量不超过 40 mg/kg/日",
        "warning": "6 月龄以下婴儿不推荐使用；脱水、肾功能异常时慎用",
    }


# --------------------------------------------------------------------------
# 工具登记表：名字 → (实现, 一句话说明)
# 说明会写进给 LLM 的 Prompt，也是 trace 里给用户看的
# --------------------------------------------------------------------------
TOOL_SPECS: dict[str, str] = {
    "egfr": "估算肾小球滤过率（CKD-EPI 2021）。参数：creatinine（数值）、"
            "creatinine_unit（mg/dL 或 umol/L，默认 mg/dL）、age、sex（male/female）",
    "bmi": "计算体重指数并给出中国成人分类。参数：weight_kg、height_cm",
    "pediatric_acetaminophen": "儿童对乙酰氨基酚按体重剂量。参数：weight_kg",
    "pediatric_ibuprofen": "儿童布洛芬按体重剂量。参数：weight_kg",
}


def run_tool(name: str, args: dict) -> dict:
    """跑一个工具，返回 {"summary": str, "data": dict}。

    任何异常（参数缺失 / 取值非法）都抛给调用方，由 `compute_for_question`
    统一按 fail-open 处理 —— 这里不吞异常，否则"算错了"和"没算"就分不清了。
    """
    if name == "egfr":
        scr = creatinine_to_mg_dl(float(args["creatinine"]),
                                  args.get("creatinine_unit", "mg/dL"))
        value = egfr_ckd_epi(scr, float(args["age"]), str(args["sex"]))
        return {
            "summary": (f"按 CKD-EPI 2021 公式估算，eGFR ≈ {value:.1f} mL/min/1.73m²，"
                        f"对应 {ckd_stage(value)}。"),
            "data": {"egfr": round(value, 1), "stage": ckd_stage(value),
                     "scr_mg_dl": round(scr, 3)},
        }
    if name == "bmi":
        value = bmi(float(args["weight_kg"]), float(args["height_cm"]))
        return {
            "summary": f"BMI ≈ {value:.1f}，按中国成人标准属于「{bmi_category(value)}」。",
            "data": {"bmi": round(value, 1), "category": bmi_category(value)},
        }
    if name == "pediatric_acetaminophen":
        data = pediatric_acetaminophen_dose(float(args["weight_kg"]))
        lo, hi = data["per_dose_mg"]
        return {
            "summary": (f"按体重 {args['weight_kg']}kg 计，对乙酰氨基酚每次 {lo}~{hi} mg，"
                        f"{data['interval_h']}；{data['max_per_day']}。"),
            "data": data,
        }
    if name == "pediatric_ibuprofen":
        data = pediatric_ibuprofen_dose(float(args["weight_kg"]))
        lo, hi = data["per_dose_mg"]
        return {
            "summary": (f"按体重 {args['weight_kg']}kg 计，布洛芬每次 {lo}~{hi} mg，"
                        f"{data['interval_h']}；{data['max_per_day']}。{data['warning']}"),
            "data": data,
        }
    raise ValueError(f"未知工具：{name}")


# --------------------------------------------------------------------------
# 规划层：让 LLM 决定"要不要算、算什么、参数是多少"
# --------------------------------------------------------------------------

_PLAN_PROMPT = """你是一个医学计算工具调度器。判断用户问题是否需要调用下面的确定性计算工具。

可用工具（名字 → 用途）：
{tools}

【规则】
1. 只有当用户**明确给出了计算所需的数值**时才选工具；缺任何一个必需参数就返回 null。
2. 只做"算"，不做诊断、不建议用药方案 —— 那不是你的事。
3. 不要从问题里"猜"参数（例如用户只说身高没说体重，就不要编体重）。
4. creatinine_unit：用户写 μmol/L 或 umol/L 时填 "umol/L"，否则 "mg/dL"。
5. sex 用 "male" / "female"。用户写"男/女"也要转过去。

【输出】只输出一个 JSON 对象，不要解释、不要代码块围栏。两种形态：
  需要计算：{{"tool": "工具名", "args": {{...}}}}
  不需要或参数不全：{{"tool": null}}

示例：
  用户：肌酐 200μmol/L，男，70 岁，eGFR 多少
  => {{"tool": "egfr", "args": {{"creatinine": 200, "creatinine_unit": "umol/L", "age": 70, "sex": "male"}}}}
  用户：孩子发烧了怎么办
  => {{"tool": null}}

用户问题：{question}
"""

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def parse_tool_plan(raw: str) -> tuple[str, dict]:
    """解析规划层输出 → (tool_name, args)。**任何异常/缺参数都返回 ("", {})**。

    与 `parse_grade_output` 同一条纪律：解析层绝不抛异常出去，绝不变成故障点。
    """
    try:
        text = _FENCE_RE.sub("", str(raw)).strip()
        # 模型偶尔会在 JSON 前后加一句话，截取第一个 {...}
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return "", {}
        obj = json.loads(text[start:end + 1])
        tool = obj.get("tool") or ""
        if tool not in TOOL_SPECS:
            return "", {}
        args = obj.get("args") or {}
        if not isinstance(args, dict):
            return "", {}
        return tool, args
    except Exception:
        return "", {}


def compute_for_question(question: str) -> dict | None:
    """整条工具链的入口：判 → 算 → 返回结果 dict；不需要或任何失败返回 None。

    返回 {"tool", "args", "summary", "data"}。
    ⚠️ 这条链上**任何一步失败都只是"没算"**，调用方按"没有工具结果"继续走原路径。
    """
    try:
        prompt = _PLAN_PROMPT.format(
            tools="\n".join(f"  {k} —— {v}" for k, v in TOOL_SPECS.items()),
            question=question,
        )
        raw = get_llm().invoke(prompt).content
        name, args = parse_tool_plan(str(raw))
        if not name:
            return None
        out = run_tool(name, args)
        return {"tool": name, "args": args, **out}
    except Exception as e:
        print(f"[工具节点] 跳过（不影响本轮问答）: {type(e).__name__}: {e}")
        return None
