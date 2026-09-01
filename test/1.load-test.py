# 加载整个数据集
from app.data_loader import load_documents

load_documents()
load_documents(limit_per_file=500)

# 加载单个文件
load_documents(path="医学指南.pdf")
load_documents(path="病历.docx")
load_documents(path="补充数据.csv", limit_per_file=100)