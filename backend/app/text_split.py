from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

#滑动窗口切分文档
def split_documents(
    documents: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Document]:
    """将长文档切分为 chunk

    Args:
        chunk_size: 每个块最大字符数（中文建议 300-800）
        chunk_overlap: 相邻块重叠字符数（建议 chunk_size 的 10%-20%）
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "；", "，", " ", ""],  # 中文优先按段落→句子切
    )
    return splitter.split_documents(documents)