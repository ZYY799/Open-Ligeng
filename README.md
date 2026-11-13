# Open Ligeng

一个本地运行的中文聊天界面和 HTTP 推理接口，加载 Qwen2.5-7B-Instruct 与第 2 轮 LoRA。模型回复是ligeng风格模拟。

需要 Python 3.10+、支持 BF16 的 NVIDIA CUDA GPU，以及能容纳 7B 权重和生成缓存的显存（建议 24 GB）。先根据系统和 CUDA 版本安装 [PyTorch](https://pytorch.org/get-started/locally/)，然后运行：

    python -m pip install -r requirements.txt
    python app.py

打开 [127.0.0.1:8765](http://127.0.0.1:8765/)。第一次启动会下载基础模型和 [LoRA 权重](https://huggingface.co/zyss1/qwen25-7b-openligeng)。

已有本地模型文件时可以离线运行：

    python app.py --base /path/to/base --adapter /path/to/adapter --offline

默认仅监听本机。部署到自己的 GPU 主机时，可以通过 SSH 端口转发打开网页：

    ssh -N -L 8765:127.0.0.1:8765 user@host

调用接口：

    curl http://127.0.0.1:8765/api/chat \
      -H 'Content-Type: application/json' \
      -d '{"messages":[{"role":"user","content":"今晚吃什么？"}]}'

接口返回含有 reply 字段的 JSON。多轮对话时由客户端传入最近的 user / assistant 消息，最后一条是 user。程序不保存对话。

仓库只包含推理代码和网页。训练数据、聊天原文和评估样本不公开。基础模型见 [Qwen 模型卡](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)。
