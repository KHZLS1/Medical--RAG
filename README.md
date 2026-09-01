# 医疗智能问答系统 (Medical RAG)

基于 **React + FastAPI + LangChain + DeepSeek + Milvus** 的智能医疗问答系统，使用开源中文医疗问答数据集（79 万条 Q&A）作为知识库。

## ✨ 特性

- 📚 **RAG 检索增强生成**：从医学知识库检索相关问答，喂给大模型生成可溯源回答
- 🤖 **DeepSeek 大模型**：调用 DeepSeek-V3 API，中文医疗理解强、成本低
- 🔍 **bge-large-zh Embedding**：中文医疗语义检索 SOTA 模型
- 💾 **Milvus 向量库**：生产级向量数据库
- ⚡ **SSE 流式输出**：前端打字机效果，回答实时增量显示
- 📎 **引用溯源**：每条回答附来源文档片段
- ⚠️ **医疗安全**：低温度防幻觉、急症提示、免责声明、拒答处方剂量

## 📁 项目结构

```
医疗RAG/
├── Chinese-medical-dialogue-data-master/   # 数据集 (GBK 编码)
│   └── Data_数据/
│       ├── Andriatria_男科/男科5-13000.csv
│       ├── IM_内科/内科5000-33000.csv
│       ├── OAGD_妇产科/妇产科6-28000.csv
│       ├── Oncology_肿瘤科/肿瘤科5-10000.csv
│       ├── Pediatric_儿科/儿科5-14000.csv
│       └── Surgical_外科/外科5-14000.csv
├── backend/                                # FastAPI + LangChain 后端
│   ├── app/
│   │   ├── main.py            # FastAPI 入口 (含 /chat /ingest /health)
│   │   ├── config.py          # 配置 (读取 .env)
│   │   ├── data_loader.py     # 医疗 CSV 加载器 (GBK 解析)
│   │   ├── vectorstore.py     # Milvus + bge Embedding 封装
│   │   ├── llm.py             # DeepSeek 客户端
│   │   └── rag_chain.py      # LangChain RAG 链 + 医疗 Prompt
│   ├── scripts/ingest.py      # 批量入库脚本
│   ├── requirements.txt
│   └── .env.example
├── frontend/                              # React + Vite + TS 前端
│   ├── src/
│   │   ├── App.tsx           # 顶层组件
│   │   ├── components/Chat.tsx  # 聊天 UI
│   │   ├── api/chat.ts       # SSE 客户端
│   │   └── index.css
│   ├── package.json
│   └── vite.config.ts
└── docker-compose.yml                      # Milvus + etcd + minio
```

## 🚀 4 步快速跑通

### 步骤 1：启动 Milvus 向量库

在项目根目录执行：

```bash
docker compose up -d
```

等待 30 秒左右，访问 http://localhost:9091 可看到 Milvus 健康面板。

### 步骤 2：配置后端

```bash
cd backend

# 创建虚拟环境
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
# Linux/macOS
# source .venv/bin/activate

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env，填入你的 DEEPSEEK_API_KEY
# 申请地址: https://platform.deepseek.com/api_keys
```

### 步骤 3：数据入库

首次入库建议限量，快速验证（每个科室 500 条 = 约 3000 条）：

```bash
# 在 backend/ 目录下
python scripts/ingest.py --limit 500
```

> 首次运行会自动下载 bge-large-zh-v1.5 模型（约 1.3GB），需联网。
> 全量入库 79 万条耗时较长，CPU 约 10+ 小时，建议有 GPU 或用云端 Embedding API。

### 步骤 4：启动服务

```bash
# 终端 1: 启动后端 (在 backend/ 目录)
uvicorn app.main:app --reload --port 8000

# 终端 2: 启动前端 (在 frontend/ 目录)
cd ../frontend
npm install
npm run dev
```

打开 http://localhost:5173 ，看到聊天界面即成功！

### 验证流程

1. 顶部状态栏应显示 `Milvus: http://localhost:19530 | collection: medical_qa`
2. 点击右上角「一键入库 500 条」按钮（若步骤 3 已入库可跳过）
3. 在输入框提问，例如：「高血压患者能吃党参吗？」
4. 助手会流式输出回答，下方显示引用来源

## 🛠️ 常见问题

### Q: 入库时报 "未配置 DEEPSEEK_API_KEY"
A: 编辑 `backend/.env`，填入你的 DeepSeek API Key。

### Q: 前端 SSE 不流式 / 一次性返回全部内容
A: 检查 `vite.config.ts` 的 proxy 配置是否指向 8000 端口，且后端 `uvicorn` 已启动。

### Q: Milvus 连接失败
A: `docker compose ps` 查看 milvus 容器状态；初次启动可能要等 30s+。

### Q: bge 模型下载慢
A: 可设置 HuggingFace 镜像：
  ```bash
  set HF_ENDPOINT=https://hf-mirror.com   # Windows PowerShell
  # export HF_ENDPOINT=https://hf-mirror.com  # Linux/macOS
  ```

### Q: CPU 跑 Embedding 太慢
A: 修改 `backend/.env` 中 `EMBEDDING_DEVICE=cuda`（需 GPU + CUDA）；或换 `BAAI/bge-small-zh-v1.5`。

## 🔒 医疗安全提示

- 本系统回答**仅供参考**，不能替代执业医师诊断
- 数据来源为开源医疗问答数据集，可能存在错漏
- 严禁用于实际诊疗决策
- 涉及急症请立即拨打 120

## 📌 后续可扩展方向

- [ ] 多轮对话上下文（会话历史 + 历史压缩）
- [ ] 混合检索 (向量 + BM25 + Reranker)
- [ ] 用户鉴权与会话持久化 (PostgreSQL)
- [ ] 文档上传与增量索引
- [ ] 评估体系 (Ragas / LangSmith)
- [ ] 医疗同义词词典扩展召回
- [ ] LangGraph Agent 主动反问
