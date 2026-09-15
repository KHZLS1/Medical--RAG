"""文档管理路由：上传、查看、删除（元数据存入数据库，文件保留磁盘）"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks
from sqlalchemy.orm import Session

from ..data_loader import (
    save_upload_file,
    list_uploaded_files,
    delete_uploaded_file,
    load_single_file,
    is_allowed_file,
)
from ..database import get_db
from ..models import UploadedDocument
from ..text_split import split_documents
from ..vectorstore import add_documents, delete_documents_by_filename

router = APIRouter(prefix="/api", tags=["文档管理"])


def _index_file(file_path: str, filename: str) -> None:
    """后台任务：加载上传文档 → 切分 → 向量化入 Milvus"""
    try:
        docs = load_single_file(file_path)
        for d in docs:
            d.metadata.setdefault("department", "上传文档")
            d.metadata.setdefault("filename", filename)
        chunks = split_documents(docs, chunk_size=500, chunk_overlap=50)
        count = add_documents(chunks)
        print(f"[索引] {filename}: {count} 个块已入库")
    except Exception as e:
        print(f"[索引错误] {filename}: {e}")


@router.post("/upload", summary="上传单个文档")
async def upload_document(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        db: Session = Depends(get_db),
):
    """上传文档：文件保存到磁盘，元数据写入数据库，后台向量化入 Milvus

    支持的文件类型: .pdf .docx .txt .md .csv
    单文件大小上限: 100MB
    同名文件不会被覆盖，会自动改名为 xxx(1).csv 之类
    """
    # 先校验扩展名，避免为不支持的格式白读一遍大文件
    if not is_allowed_file(file.filename or ""):
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: {file.filename}")

    content = await file.read()

    # ===== 去重：相同内容直接拒绝，不写盘不入库 =====
    import hashlib
    digest = hashlib.sha256(content).hexdigest()
    dup = db.query(UploadedDocument).filter(UploadedDocument.content_hash == digest).first()
    if dup:
        raise HTTPException(
            status_code=409,
            detail=f"已上传过相同内容的文件: {dup.filename}"
        )

    # 1. 保存文件到磁盘
    try:
        file_path = save_upload_file(content, file.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 2. 元数据写入数据库（filename 用实际落盘名，可能已改名）
    import os
    stored_name = os.path.basename(file_path)
    doc_record = UploadedDocument(
        filename=stored_name,
        file_size=len(content),
        file_type=os.path.splitext(stored_name)[1].lower(),
        file_path=str(file_path),
        content_hash=digest,
    )
    db.add(doc_record)
    db.commit()
    db.refresh(doc_record)

    # 3. 后台切分 + 向量化入 Milvus（不阻塞上传响应）
    #    metadata 里的 filename 必须与数据库记录一致，删除时才能按它清理向量
    background_tasks.add_task(_index_file, str(file_path), stored_name)

    return {
        "id": doc_record.id,
        "filename": doc_record.filename,
        "saved_path": str(file_path),
        "size_bytes": len(content),
        "size_mb": round(len(content) / (1024 * 1024), 2),
        "file_type": doc_record.file_type,
        "uploaded_at": doc_record.uploaded_at.isoformat() if doc_record.uploaded_at else None,
        "renamed": stored_name != (file.filename or ""),
        "indexing": "后台向量化索引中",
    }
@router.get("/uploads",summary="列出已上传文档")
async def list_uploads(db: Session = Depends(get_db)):
    """从数据库查询所有已上传文档"""
    docs = db.query(UploadedDocument).order_by(UploadedDocument.uploaded_at.desc()).all()
    return {
        "files": [
            {
                "id": d.id,
                "filename": d.filename,
                "size_bytes": d.file_size,
                "size_mb": round(d.file_size / (1024 * 1024), 2),
                "extension": d.file_type,
                "file_path": d.file_path,
                "uploaded_at": d.uploaded_at.isoformat() if d.uploaded_at else None,
            }
            for d in docs
        ]
    }

@router.delete("/uploads/{doc_id}", summary="删除指定上传文档")
async def remove_upload(
        doc_id: int,
        db: Session = Depends(get_db)
):
    """删除文档：按主键删除，同时清理磁盘文件、数据库记录和已入库的向量

    说明：原先按 filename 删除，但文件名并不唯一（同名上传会被改名，
    历史数据也可能重名），会删错记录，因此改为按 id 精确删除。
    """
    doc = db.query(UploadedDocument).filter(UploadedDocument.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail=f"文档记录不存在: id={doc_id}")

    filename = doc.filename

    # 1. 删除磁盘文件（文件可能已被手工删掉，不算失败）
    file_deleted = delete_uploaded_file(filename)

    # 2. 删除数据库记录
    db.delete(doc)
    db.commit()

    # 3. 清理该文档已入库的向量（失败不影响删除结果，但要如实返回）
    vector_error = None
    try:
        deleted_vectors = delete_documents_by_filename(filename)
    except Exception as e:
        deleted_vectors = 0
        vector_error = f"{type(e).__name__}: {e}"

    return {
        "deleted": doc_id,
        "filename": filename,
        "file_removed": file_deleted,
        "vectors_deleted": deleted_vectors,
        "vector_error": vector_error,
    }