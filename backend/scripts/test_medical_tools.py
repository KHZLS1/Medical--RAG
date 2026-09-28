"""离线自测：确定性医学计算工具（T62-⑤）

只验证**纯函数算得对**、解析层**不吃畸形输入**、整条链**失败即跳过**。
全程不连 Milvus / MySQL / LLM（LLM 用假对象替换）。

跑法：python scripts/test_medical_tools.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import medical_tools as mt
from app.medical_tools import (
    bmi, bmi_category, ckd_stage, compute_for_question, creatinine_to_mg_dl,
    egfr_ckd_epi, parse_tool_plan, run_tool,
)

_fails: list[str] = []


def check(name: str, cond, detail: str = "") -> None:
    ok = bool(cond)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail and not ok else ""))
    if not ok:
        _fails.append(name)


def near(a: float, b: float, tol: float = 0.15) -> bool:
    return abs(a - b) <= tol


# ---------------------------------------------------------------- 单位换算
print("== 1. 肌酐单位换算（漏换会让 eGFR 差 ~10 倍）==")
check("200 μmol/L → 2.26 mg/dL", near(creatinine_to_mg_dl(200, "μmol/L"), 2.2624, 0.001))
check("200 umol/l 小写也认", near(creatinine_to_mg_dl(200, "umol/l"), 2.2624, 0.001))
check("mg/dL 原样返回", creatinine_to_mg_dl(1.2, "mg/dL") == 1.2)
check("单位缺省按 mg/dL", creatinine_to_mg_dl(1.2) == 1.2)

# ---------------------------------------------------------------- eGFR
print("\n== 2. eGFR（CKD-EPI 2021，无种族项）==")
# 手算基准：142 × min(Scr/κ,1)^α × max(Scr/κ,1)^-1.2 × 0.9938^age (×1.012 女)
check("男 70 岁 / Scr 1.0 → ≈81.0", near(egfr_ckd_epi(1.0, 70, "male"), 81.0))
check("女 60 岁 / Scr 1.0 → ≈64.5", near(egfr_ckd_epi(1.0, 60, "female"), 64.5))
check("男 70 岁 / Scr 2.26 → ≈30.4", near(egfr_ckd_epi(2.2624, 70, "male"), 30.4))
check("中文性别「男」也能识别", near(egfr_ckd_epi(1.0, 70, "男"), 81.0))
# ⚠️ 这里曾写反：女性系数 1.012 是**上调**，但 κ(0.7 vs 0.9) 与 α 的差异主导，
# 净效果是**同 Scr 同年龄下女性 eGFR 更低**（女性肌肉量少，同 Scr 意味着更差的肾功能）。
# 断言要按临床事实写，不能按"看到 1.012 就以为女性更高"。
check("同 Scr 同年龄：女性 eGFR 低于男性（κ/α 差异主导）",
      egfr_ckd_epi(1.0, 70, "female") < egfr_ckd_epi(1.0, 70, "male"))
check("肌酐越低 eGFR 越高（单调性）",
      egfr_ckd_epi(0.8, 50, "male") > egfr_ckd_epi(2.0, 50, "male"))
try:
    egfr_ckd_epi(1.0, 70, "未知")
    check("性别无法识别 → 抛错（不静默算个错的）", False)
except ValueError:
    check("性别无法识别 → 抛错（不静默算个错的）", True)
try:
    egfr_ckd_epi(0, 70, "male")
    check("肌酐为 0 → 抛错", False)
except ValueError:
    check("肌酐为 0 → 抛错", True)

# ---------------------------------------------------------------- CKD 分期
print("\n== 3. CKD 分期边界 ==")
for val, want in ((95, "G1"), (90, "G1"), (89.9, "G2"), (60, "G2"),
                  (59.9, "G3a"), (45, "G3a"), (44.9, "G3b"), (30, "G3b"),
                  (29.9, "G4"), (15, "G4"), (14.9, "G5")):
    check(f"eGFR {val} → {want}", ckd_stage(val).startswith(want), ckd_stage(val))

# ---------------------------------------------------------------- BMI
print("\n== 4. BMI 与中国成人分类 ==")
check("70kg / 175cm → 22.9", near(bmi(70, 175), 22.86, 0.01))
check("中国标准 <18.5 偏瘦", bmi_category(18.49) == "偏瘦")
check("中国标准 18.5 正常", bmi_category(18.5) == "正常")
check("中国标准 24 超重（WHO 是 25）", bmi_category(24.0) == "超重")
check("中国标准 28 肥胖（WHO 是 30）", bmi_category(28.0) == "肥胖")
check("27.9 仍是超重", bmi_category(27.9) == "超重")

# ---------------------------------------------------------------- 儿童剂量
print("\n== 5. 儿童按体重剂量 ==")
ap = mt.pediatric_acetaminophen_dose(30)
check("对乙酰氨基酚 30kg → 300~450 mg", ap["per_dose_mg"] == (300.0, 450.0), str(ap["per_dose_mg"]))
check("对乙酰氨基酚间隔 4~6h", "4~6" in ap["interval_h"])
ib = mt.pediatric_ibuprofen_dose(20)
check("布洛芬 20kg → 100~200 mg", ib["per_dose_mg"] == (100.0, 200.0), str(ib["per_dose_mg"]))
check("布洛芬带 6 月龄以下警示", "6 月龄" in ib.get("warning", ""))

# ---------------------------------------------------------------- run_tool
print("\n== 6. run_tool 的 summary 可直接给用户看 ==")
r = run_tool("egfr", {"creatinine": 200, "creatinine_unit": "umol/L", "age": 70, "sex": "male"})
check("summary 含 eGFR 数值", "eGFR" in r["summary"] and "30" in r["summary"], r["summary"])
check("summary 含分期", "G3b" in r["summary"], r["summary"])
check("data 里带上归一后的肌酐", near(r["data"]["scr_mg_dl"], 2.262, 0.001))
r2 = run_tool("bmi", {"weight_kg": 70, "height_cm": 175})
check("BMI summary 含分类", "正常" in r2["summary"], r2["summary"])
try:
    run_tool("egfr", {"age": 70, "sex": "male"})       # 缺 creatinine
    check("缺参数 → 抛错（交给上层 fail-open）", False)
except Exception:
    check("缺参数 → 抛错（交给上层 fail-open）", True)
try:
    run_tool("不存在的工具", {})
    check("未知工具名 → 抛错", False)
except ValueError:
    check("未知工具名 → 抛错", True)

# ---------------------------------------------------------------- 解析层
print("\n== 7. parse_tool_plan 吃畸形输入（绝不抛异常）==")
ok_plan = '{"tool": "egfr", "args": {"creatinine": 1.0, "age": 70, "sex": "male"}}'
name, args = parse_tool_plan(ok_plan)
check("正常 JSON 解析成功", name == "egfr" and args["age"] == 70)
name, _ = parse_tool_plan(f"```json\n{ok_plan}\n```")
check("``` 围栏能剥掉", name == "egfr")
name, _ = parse_tool_plan(f"好的，结果如下：\n{ok_plan}\n以上。")
check("前后有杂话也能截出 JSON", name == "egfr")
check("tool 为 null → 空", parse_tool_plan('{"tool": null}') == ("", {}))
check("tool 不在登记表 → 空", parse_tool_plan('{"tool": "shell", "args": {}}') == ("", {}))
check("args 不是对象 → 空", parse_tool_plan('{"tool": "egfr", "args": "x"}') == ("", {}))
check("非法 JSON → 空", parse_tool_plan("{不是 json") == ("", {}))
check("空串 → 空", parse_tool_plan("") == ("", {}))
check("None → 空", parse_tool_plan(None) == ("", {}))
check("没有任何花括号 → 空", parse_tool_plan("我只是说句话") == ("", {}))

# ---------------------------------------------------------------- 整链 fail-open
print("\n== 8. compute_for_question：失败即「没算」，绝不拦问答 ==")
_real_get_llm = mt.get_llm


class _FakeLLM:
    def __init__(self, content=None, boom=False):
        self._content, self._boom = content, boom

    def invoke(self, _prompt):
        if self._boom:
            raise RuntimeError("provider 挂了")
        return type("R", (), {"content": self._content})()


mt.get_llm = lambda *a, **k: _FakeLLM(
    '{"tool": "bmi", "args": {"weight_kg": 70, "height_cm": 175}}')
out = compute_for_question("我 70 公斤 175 厘米，BMI 多少")
check("命中工具 → 返回结果", out is not None and out["tool"] == "bmi", str(out))
check("结果带 summary", out and "BMI" in out["summary"])

mt.get_llm = lambda *a, **k: _FakeLLM('{"tool": null}')
check("LLM 说不需要 → None", compute_for_question("孩子发烧了怎么办") is None)

mt.get_llm = lambda *a, **k: _FakeLLM(boom=True)
check("LLM 抛异常 → None（不影响本轮问答）", compute_for_question("肌酐多少") is None)

mt.get_llm = lambda *a, **k: _FakeLLM('{"tool": "egfr", "args": {"age": 70}}')
check("参数不全 → None（不拿残缺参数硬算）", compute_for_question("我 70 岁") is None)

mt.get_llm = _real_get_llm

# ---------------------------------------------------------------- 契约
print("\n== 9. 工具登记表与免责声明 ==")
check("登记了 4 个工具", len(mt.TOOL_SPECS) == 4, str(list(mt.TOOL_SPECS)))
check("每个工具都有说明文字", all(v.strip() for v in mt.TOOL_SPECS.values()))
check("免责声明提到「不能替代医生」", "不能替代医生" in mt.DISCLAIMER)

print()
if _fails:
    print(f"❌ {len(_fails)} 项未通过：{_fails}")
    sys.exit(1)
print("全绿：医学工具（纯函数 / 解析层 / fail-open / 登记表）均通过。")
sys.exit(0)
