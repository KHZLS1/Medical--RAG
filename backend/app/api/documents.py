"""文档管理路由：上传、查看、删除（元数据存入数据库，文件保留磁盘）"""
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from sqlalchemy.orm import Session

from ..data_loader import (
    save_upload_file,
    list_uploaded_files,
    delete_uploaded_file,
)
from ..database import get_db
from ..models import UploadedDocument

router = APIRouter(prefix="/api", tags=["文档管理"])

@router.post("/upload", summary="上传单个文档")
async def upload_document(
        file: UploadFile = File(...),
        db: Session = Depends(get_db),
):
    """上传文档：文件保存到磁盘，元数据写入数据库

    支持的文件类型: .pdf .docx .txt .md .csv
    单文件大小上限: 100MB
    """
    content = await file.read()

    # 1. 保存文件到磁盘
    try:
        file_path = save_upload_file(content, file.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # 2. 元数据写入数据库
    import os
    doc_record = UploadedDocument(
        filename=os.path.basename(file_path),
        file_size=len(content),
        file_type=os.path.splitext(file.filename)[1].lower(),
        file_path=str(file_path),
    )
    db.add(doc_record)
    db.commit()
    db.refresh(doc_record)

    return {
        "id": doc_record.id,
        "filename": doc_record.filename,
        "saved_path": str(file_path),
        "size_bytes": len(content),
        "size_mb": round(len(content) / (1024 * 1024), 2),
        "file_type": doc_record.file_type,
        "uploaded_at": doc_record.uploaded_at.isoformat() if doc_record.uploaded_at else None,
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

@router.delete("/uploads/{filename}",summary="删除指定上传文档")
async def remove_upload(
        filename: str,
        db: Session = Depends(get_db)
):
    """删除文档：同时删除磁盘文件和数据库记录"""
    # 1. 删除数据库记录
    doc = db.query(UploadedDocument).filter(UploadedDocument.filename == filename).first()
    if not doc:
        raise HTTPException(status_code=404, detail=f"文件不存在: {filename}")

    # 2. 删除磁盘文件
    deleted = delete_uploaded_file(filename)
    if not deleted:
        raise HTTPException(status_code=404,detail=f"文件不存在或不可删除: {filename}",)

    # 3. 删除数据库记录
    db.delete(doc)
    db.commit()

    return {"deleted": filename}