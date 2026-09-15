"""全局配置：从 .env 读取，pydantic-settings 管理类型与默认值"""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    #上传文档
    upload_dir: str = "uploads"  # 上传文件目录 (相对 backend)
    upload_max_size_mb: int = 100  # 单文件大小上限
    upload_allowed_extensions: str = ".pdf,.docx,.txt,.md,.csv"  # 允许的扩展名

    # DeepSeek
    deepseek_api_key: str = ""

    # Milvus
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection: str = "medical_qa"

    # Embedding
    embedding_model: str = "BAAI/bge-large-zh-v1.5"
    embedding_device: str = "cpu"

    #查询改写
    query_rewrite_enabled: bool = True

    # 数据库（在 .env 中配置 DATABASE_URL，勿在代码里写明文密码）
    database_url: str = ""

    # 数据路径
    medical_data_dir: str = "../Chinese-medical-dialogue-data-master/Data_数据"
    cleaned_data_dir: str = "data/by_department"

    # 检索权重（eval_rag.py tune 可搜出最优值，写进 .env 或 tuned 文件生效）
    bm25_weight: float = 0.3
    vector_weight: float = 0.7
    reranker_top_k: int = 5

    # tune 输出最优权重的落盘路径（后端自动叠加到默认值）
    tuned_weights_path: str = "data/tuned_weights.json"
    
    # 服务
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    frontend_url: str = "http://localhost:5173"

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    @property
    def data_dir_resolved(self) -> Path:
        """返回数据集绝对路径（相对 backend 目录解析）"""
        return Path(__file__).resolve().parent.parent / self.medical_data_dir

    @property
    def cleaned_data_dir_resolved(self) -> Path:
        """返回清洗后数据的绝对路径"""
        return Path(__file__).resolve().parent.parent / self.cleaned_data_dir

    @property
    def milvus_uri(self) -> str:
        return f"http://{self.milvus_host}:{self.milvus_port}"

    @property
    def upload_dir_resolved(self) -> Path:
        """返回上传目录绝对路径"""
        return Path(__file__).resolve().parent.parent / self.upload_dir


settings = Settings()
