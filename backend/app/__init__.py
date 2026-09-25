"""app 包初始化：在任何 transformers / huggingface_hub 子模块被导入之前，
打开离线开关，阻止进程向 huggingface.co 发请求校验模型版本。

为什么必须放在这里：huggingface_hub 的 HF_HUB_OFFLINE 是**模块级常量**，
import 时即定型。若等到 config.py 或 reranker.py 里再设 os.environ，已经晚了
——那时 transformers/huggingface_hub 早被导入，常量不会重读。
本文件是 `import app.xxx` 时最先执行的模块，是代码层唯一可靠的时机。

国内直连 huggingface.co 会在 TLS 阶段被阻断（SSLEOFError），
因此离线是默认行为。用 setdefault 而非直接赋值，保留显式覆写的余地；
如需临时联网（例如下载新模型），把环境变量置 0 并重启进程即可。

PyCharm Run/Debug 配置里再配一份同名环境变量作为双保险。
"""
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
