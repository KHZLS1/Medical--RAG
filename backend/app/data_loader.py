"""医疗文档加载器

支持三种加载模式:
  1. 批量加载医疗问答数据集（默认，扫描整个数据目录）
  2. 加载单个指定文件（PDF/Word/TXT/Markdown/CSV）
  3. 加载用户上传的文档

数据集结构:
  Data_数据/
    ├─ Andriatria_男科/男科5-13000.csv
    ├─ IM_内科/内科5000-33000.csv
    ├─ OAGD_妇产科/妇产科6-28000.csv
    ├─ Oncology_肿瘤科/肿瘤科5-10000.csv
    ├─ Pediatric_儿科/儿科5-14000.csv
    └─ Surgical_外科/外科5-14000.csv
"""
import csv
import time
from pathlib import Path
from typing import Iterator

from langchain_core.documents import Document

from .config import settings

# ============================================================
# 上传目录管理（扩展名与大小上限来自 .env，见 config.py）
# ============================================================
UPLOAD_DIR = Path(__file__).resolve().parent.parent / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# 支持的文件扩展名，如 UPLOAD_ALLOWED_EXTENSIONS=".pdf,.docx,.txt,.md,.csv"
ALLOWED_EXTENSIONS = {
    ext.strip().lower()
    for ext in settings.upload_allowed_extensions.split(",")
    if ext.strip()
}

# 单个文件大小上限
MAX_FILE_SIZE = settings.upload_max_size_mb * 1024 * 1024

def get_upload_dir() -> Path:
    """返回上传文件存储目录"""
    return UPLOAD_DIR

def is_allowed_file(filename: str) -> bool:
    """检查文件扩展名是否在允许列表中"""
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS

def save_upload_file(file_bytes: bytes, filename: str) -> Path:
    """保存上传的文件到 uploads 目录

    同名文件不会被覆盖：目标名已存在时自动追加 (1)(2)... 后缀。

    Args:
        file_bytes: 文件二进制内容
        filename: 原始文件名

    Returns:
        保存后的文件绝对路径（文件名可能与原始名不同）

    Raises:
        ValueError: 文件类型不允许 / 文件过大 / 文件名含非法字符
    """
    if not is_allowed_file(filename):
        raise ValueError(
            f"不支持的文件类型 '{filename}'，仅支持: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    if len(file_bytes) > MAX_FILE_SIZE:
        raise ValueError(
            f"文件过大 ({len(file_bytes) // (1024 * 1024)}MB)，最大支持 {MAX_FILE_SIZE // (1024 * 1024)}MB"
        )

    # 防止路径穿越攻击
    safe_name = Path(filename).name
    if not safe_name or safe_name.startswith("."):
        raise ValueError(f"非法文件名: {filename}")
    # 引号/反斜杠会破坏 Milvus JSON 过滤表达式（删除向量时要用文件名做 filter）
    if any(ch in safe_name for ch in ('"', "'", "\\")):
        raise ValueError(f"文件名含非法字符（引号或反斜杠）: {filename}")

    file_path = UPLOAD_DIR / safe_name

    # 同名不覆盖：追加 (1)(2)... 直到找到空闲名字
    if file_path.exists():
        stem, suffix = Path(safe_name).stem, Path(safe_name).suffix
        for index in range(1, 1000):
            candidate = UPLOAD_DIR / f"{stem}({index}){suffix}"
            if not candidate.exists():
                file_path = candidate
                break
        else:
            raise ValueError(f"同名文件过多，无法为 {safe_name} 生成新文件名")

    file_path.write_bytes(file_bytes)
    return file_path

def list_uploaded_files() -> list[dict]:
    """列出 uploads 目录下所有已上传文件"""
    files = []
    for f in sorted(UPLOAD_DIR.iterdir()):
        if f.is_file() and is_allowed_file(f.name):
            stat = f.stat()
            files.append({
                "filename": f.name,
                "size_bytes": stat.st_size,
                "size_mb": round(stat.st_size / (1024 * 1024), 2),
                "extension": f.suffix.lower(),
                "uploaded_at": stat.st_mtime,
            })
    return files

def delete_uploaded_file(filename: str) -> bool:
    """删除已上传的文件，返回是否删除成功"""
    safe_name = Path(filename).name
    file_path = UPLOAD_DIR / safe_name
    if file_path.exists() and file_path.is_file() and is_allowed_file(safe_name):
        file_path.unlink()
        return True
    return False

# ============================================================
# 单文件加载（PDF / Word / TXT / Markdown / CSV）
# ============================================================
def load_single_file(file_path: str | Path) -> list[Document]:
    """加载单个文件，返回 LangChain Document 列表

    根据扩展名自动选择对应的 loader:
      .pdf  -> PyPDFLoader
      .docx -> Docx2txtLoader
      .txt  -> TextLoader
      .md   -> TextLoader
      .csv  -> CSVLoader
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {path}")

    ext = path.suffix.lower()

    loaders = {
        ".pdf": _load_pdf,
        ".docx": _load_docx,
        ".txt": _load_text,
        ".md": _load_markdown,
        ".csv": _load_csv,
    }

    loader_fn = loaders.get(ext)
    if loader_fn is None:
        raise ValueError(f"不支持的文件类型: {ext}")

    return loader_fn(path)

def _load_pdf(path: Path) -> list[Document]:
    from langchain_community.document_loaders import PyPDFLoader
    loader = PyPDFLoader(str(path))
    docs = loader.load()
    for doc in docs:
        doc.metadata.setdefault("source", str(path))
        doc.metadata.setdefault("file_type", "pdf")
        doc.metadata.setdefault("filename", path.name)
    return docs

def _load_docx(path: Path) -> list[Document]:
    from langchain_community.document_loaders import Docx2txtLoader
    loader = Docx2txtLoader(str(path))
    docs = loader.load()
    for doc in docs:
        doc.metadata.setdefault("source", str(path))
        doc.metadata.setdefault("file_type", "docx")
        doc.metadata.setdefault("filename", path.name)
    return docs

def _load_text(path: Path) -> list[Document]:
    from langchain_community.document_loaders import TextLoader
    loader = TextLoader(str(path), encoding="utf-8")
    docs = loader.load()
    for doc in docs:
        doc.metadata.setdefault("source", str(path))
        doc.metadata.setdefault("file_type", "txt")
        doc.metadata.setdefault("filename", path.name)
    return docs

def _load_markdown(path: Path) -> list[Document]:
    from langchain_community.document_loaders import TextLoader
    loader = TextLoader(str(path), encoding="utf-8")
    docs = loader.load()
    for doc in docs:
        doc.metadata.setdefault("source", str(path))
        doc.metadata.setdefault("file_type", "markdown")
        doc.metadata.setdefault("filename", path.name)
    return docs

def _load_csv(path: Path) -> list[Document]:
    from langchain_community.document_loaders import CSVLoader
    loader = CSVLoader(str(path), encoding="utf-8")
    docs = loader.load()
    for doc in docs:
        doc.metadata.setdefault("source", str(path))
        doc.metadata.setdefault("file_type", "csv")
        doc.metadata.setdefault("filename", path.name)
    return docs

# ============================================================
# 上传文档批量加载
# ============================================================
def load_uploaded_documents() -> list[Document]:
    """加载 uploads 目录下所有已上传文档

    遍历 uploads 目录，对每个支持的文件调用 load_single_file。
    加载失败的文件会打印警告并跳过，不会中断整体流程。
    """
    docs: list[Document] = []
    for f in sorted(UPLOAD_DIR.iterdir()):
        if not f.is_file() or not is_allowed_file(f.name):
            continue
        try:
            file_docs = load_single_file(f)
            docs.extend(file_docs)
            print(f"[加载] {f.name} -> {len(file_docs)} 个文档块")
        except Exception as e:
            print(f"[警告] 加载 {f.name} 失败: {e}")
    return docs

# ============================================================
# 批量加载医疗问答数据集（CSV）
# ============================================================
def iter_medical_documents(limit_per_file: int | None = None) -> Iterator[Document]:
    """迭代式加载医疗问答 CSV 数据集（节省内存）

    Args:
        limit_per_file: 每个科室 CSV 文件最多加载多少条，None 表示全部

    Yields:
        Document: 每条问答记录转为一个 Document
    """
    data_dir = settings.data_dir_resolved
    if not data_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_dir}")

    csv_files = list(data_dir.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"数据目录下未找到 CSV 文件: {data_dir}")

    for csv_path in sorted(csv_files):
        department = _parse_department(csv_path)
        yield from _iter_csv_rows(csv_path, department, limit_per_file)

def _parse_department(csv_path: Path) -> str:
    """从路径解析科室名称，如 IM_内科 -> 内科"""
    dir_name = csv_path.parent.name
    # 目录名格式: "Andriatria_男科" / "IM_内科" 等
    if "_" in dir_name:
        return dir_name.split("_", 1)[1]
    return dir_name

def _iter_csv_rows(
    csv_path: Path,
    department: str,
    limit_per_file: int | None,
) -> Iterator[Document]:
    """逐行读取 CSV，将问答对转为 Document"""
    with open(csv_path, "r", encoding="gb18030", newline="") as f:
        reader = csv.DictReader(f)
        count = 0
        for row in reader:
            if limit_per_file is not None and count >= limit_per_file:
                break

            title = (row.get("title") or "").strip()
            ask = (row.get("ask") or "").strip()
            answer = (row.get("answer") or "").strip()

            if not ask and not answer:
                continue

            # 组装文档内容
            parts = []
            if title:
                parts.append(f"标题：{title}")
            parts.append(f"问题：{ask}")
            parts.append(f"回答：{answer}")
            content = "\n".join(parts)

            # 跳过超长文本（Milvus VarChar 上限 65535 字节，超长多为脏数据）
            if len(content.encode("utf-8")) > 60000:
                continue

            yield Document(
                page_content=content,
                metadata={
                    "department": department,
                    "title": title,
                    "source": str(csv_path),
                },
            )
            count += 1

def load_medical_documents(limit_per_file: int | None = None) -> list[Document]:
    """一次性加载所有医疗问答文档到列表

    适用于小批量入库（main.py 的 /api/ingest 接口）。
    大批量请用 iter_medical_documents 流式处理（scripts/ingest.py）。
    """
    return list(iter_medical_documents(limit_per_file=limit_per_file))

def iter_cleaned_documents(limit_per_file: int | None = None) -> Iterator[Document]:
    """迭代式加载清洗后的 JSON 数据（按科室分文件）

    读取 backend/data/by_department/ 下的 JSON 文件，
    将 instruction/input/output 格式转换为 LangChain Document。

    Args:
        limit_per_file: 每个文件最多加载多少条，None 表示全部

    Yields:
        Document: 每条记录转为一个 Document
    """
    data_dir = settings.cleaned_data_dir_resolved
    if not data_dir.exists():
        raise FileNotFoundError(f"清洗数据目录不存在: {data_dir}")

    json_files = list(data_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"清洗数据目录下未找到 JSON 文件: {data_dir}")

    for json_path in sorted(json_files):
        department = json_path.stem.replace("cleaned_", "")
        yield from _iter_json_rows(json_path, department, limit_per_file)

def _iter_json_rows(
    json_path: Path,
    department: str,
    limit_per_file: int | None,
) -> Iterator[Document]:
    """逐条读取 JSON 数组，将指令格式转为 Document"""
    import json

    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    count = 0
    for rec in records:
        if limit_per_file is not None and count >= limit_per_file:
            break

        # 字段映射: instruction->title, input->ask, output->answer
        title = (rec.get("instruction") or "").strip()
        ask = (rec.get("input") or "").strip()
        answer = (rec.get("output") or "").strip()

        if not ask and not answer:
            continue

        # 组装文档内容（与 CSV 路径保持一致）
        parts = []
        if title:
            parts.append(f"标题：{title}")
        parts.append(f"问题：{ask}")
        parts.append(f"回答：{answer}")
        content = "\n".join(parts)

        # 跳过超长文本
        if len(content.encode("utf-8")) > 60000:
            continue

        yield Document(
            page_content=content,
            metadata={
                "department": department,
                "title": title,
                "source": str(json_path),
            },
        )
        count += 1