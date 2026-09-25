"""来源路由 Agent：判断问题应查通用医疗库还是上传文档，并按来源过滤检索候选

设计要点
  - 复用现有 metadata 约定：上传文档的 department=="上传文档" 且带 filename；
    通用库 department 为疾病/科室名。
  - 用一次 LLM 结构化输出做判定，失败一律回退 "auto"（不过滤），保证检索不中断。
  - use_filter 由置信度阈值控制：低置信度不加过滤，避免误判把正确文档滤掉。
"""
import json
from pydantic import BaseModel, Field
from ..llm import get_llm
from ..models import UploadedDocument
from ..database import SessionLocal

class RoutingResult(BaseModel):
    source: str = Field(..., description='来源："uploaded" | "general" | "auto"')
    confidence: float = Field(..., ge=0, le=1, description="路由置信度 0~1")
    filename: str | None = Field(default=None, description="命中的上传文件名（仅 source=uploaded 时）")
    reason: str = Field(default="", description="判定依据，一句话")

_CONF_THRESHOLD = 0.6 # 低于该置信度不加过滤，回退 auto

def _list_uploaded_filenames () -> list [str]:
    """从数据库读取当前已入库的上传文档文件名，作为 LLM 的候选来源提示"""
    db = SessionLocal()
    try:
        rows = db.query(UploadedDocument).all()
        return [r.filename for r in rows]
    finally:
        db.close()

_ROUTING_PROMPT = """你是一个知识来源路由助手。系统有两类可检索资料：
1. 通用医疗库：公开的医疗问答（按科室/疾病划分）
2. 用户上传文档：{filenames}（若为空则没有上传文档）

判断用户问题【优先查哪类资料】：
- 若问题涉及上传文档里可能包含的私有/工作资料内容，选 uploaded 并给出最可能命中的文件名
- 若是一般疾病/症状/用药咨询，选 general
- 无法判断时选 auto

只输出 JSON，不要解释。格式：
{{"source": "uploaded|general|auto", "confidence": 0.0~1.0, "filename": "xxx.csv 或 null", "reason": "一句话"}}

用户问题：{question}
"""

def route_knowledge_source (question: str) -> RoutingResult:
    """判定来源。返回 RoutingResult；任何异常都回退到 auto。"""
    # 没有任何上传文档时，直接短路为 general，省一次 LLM 调用
    filenames = _list_uploaded_filenames()
    if not filenames:
        # 注：仍有可能是"上传文档已入库但记录被删"的情况，交给过滤时兜底
        return RoutingResult(source="auto", confidence=0.5, filename=None)
    try:
        llm = get_llm()
        prompt_text = _ROUTING_PROMPT.format(filenames="、".join(filenames), question=question)
        # DeepSeek 不支持 with_structured_output 的 response_format，改为手动解析 JSON
        content = llm.invoke(prompt_text).content
        raw = content.strip()
        if raw.startswith("```"):
            raw = raw.strip("` ")
        if raw.startswith("json"):
            raw = raw[4:].lstrip()
        result = RoutingResult.model_validate(json.loads(raw))
        if result.source not in ("uploaded", "general", "auto"):
            return RoutingResult(source="auto", confidence=0.5, filename=None)
        return result
    except Exception as e:
        print(f"[来源路由] 失败，回退 auto: {type(e).__name__}: {e}")
        return RoutingResult(source="auto", confidence=0.5, filename=None)

def filter_by_source(
    docs: list,
    routing: RoutingResult,
) -> list:
    """按来源过滤检索候选。低置信度/auto 时不过滤。

        过滤规则：
          - source=uploaded   -> 只能保留 department=="上传文档" 的（可再限定 filename）
          - source=general    -> 排除 department=="上传文档" 的
          - 其他/低置信        -> 不过滤（返回原样）
        """
    # 安全阀：置信度不足或 source 无效时，绝不加过滤
    if routing.source not in ("uploaded", "general") or routing.confidence < _CONF_THRESHOLD:
        return docs

    if routing.source == "uploaded":
        kept = [d for d in docs if d.metadata.get("department") == "上传文档"]
        if routing.filename:
            # 指定了文件名就进一步收窄——但只在仍非空时收窄，避免一步到位滤光
            narrowed = [d for d in kept if d.metadata.get( "filename" ) == routing.filename]
            kept = narrowed if narrowed else kept
        return kept

    # source == "general"
    return [d for d in docs if d.metadata.get( "department" ) != "上传文档" ]