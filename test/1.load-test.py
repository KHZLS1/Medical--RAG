# 加载整个数据集（一次性载入内存，仅适合小批量验证）
from app.data_loader import load_medical_documents

load_medical_documents()
load_medical_documents(limit_per_file=500)

# 加载单个文件（按扩展名自动选择 PDF/Word/TXT/MD/CSV loader）
from app.data_loader import load_single_file

load_single_file("医学指南.pdf")
load_single_file("病历.docx")
load_single_file("补充数据.csv")