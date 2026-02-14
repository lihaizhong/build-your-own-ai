# Disney RAG问答助手 - 故障排查指南

本文档记录了在使用过程中可能遇到的常见问题及其解决方案。

## CLIP 模型下载问题

### 问题描述

在 AutoDL 平台上运行时，CLIP 模型下载失败：

```
2026-02-13 00:14:33 | INFO     | 加载CLIP模型: clip-vit-base-patch32
'[Errno 99] Cannot assign requested address' thrown while requesting HEAD https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/config.json
Retrying in 1s [Retry 1/5]. 
```

### 错误分析

**Errno 99 (EADDRNOTAVAIL)** 表示系统无法分配请求的地址，原因包括：

1. **网络无法访问 Hugging Face** - Hugging Face 服务器在国外，国内访问受限
2. **DNS 解析问题** - 无法正确解析 `huggingface.co` 域名
3. **代理配置问题** - 如果配置了代理但代理不可用

CLIP 模型（`clip-vit-base-patch32`）托管在 Hugging Face Hub 上，代码尝试从 `huggingface.co` 下载模型配置文件时连接失败。

### 解决方案

#### 方案 1：使用 HF 镜像源（推荐）

设置环境变量使用 Hugging Face 镜像：

```bash
# 临时设置（当前终端会话）
export HF_ENDPOINT=https://hf-mirror.com

# 然后再运行程序
python -m code.main --build
```

或永久设置（添加到 `~/.bashrc`）：

```bash
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
source ~/.bashrc
```

#### 方案 2：使用 AutoDL 内置镜像源

AutoDL 平台通常会预配置好 Hugging Face 镜像，检查环境变量：

```bash
# 查看是否已设置镜像
echo $HF_ENDPOINT

# 如果没有设置，手动设置
export HF_ENDPOINT=https://hf-mirror.com
```

#### 方案 3：检查 AutoDL 预置模型缓存

AutoDL 的许多镜像已经预装了常用模型：

```bash
# 查看是否有预置的模型缓存
ls -la /root/.cache/huggingface/
ls -la /root/autodl-tmp/

# 查看环境变量
env | grep -i hf
env | grep -i model
```

#### 方案 4：从 AutoDL 数据盘加载模型

如果已经将模型下载到了 AutoDL 的数据盘（`/root/autodl-tmp/`）：

```python
# 在代码中指定本地模型路径
from transformers import CLIPModel

model = CLIPModel.from_pretrained("/root/autodl-tmp/models/clip-vit-base-patch32")
```

#### 方案 5：使用 ModelScope 下载

AutoDL 平台通常访问 ModelScope 比较顺畅：

```python
from modelscope import snapshot_download

# 下载到 autodl-tmp 目录（数据盘，不会被清空）
model_dir = snapshot_download(
    'AI-ModelScope/clip-vit-base-patch32',
    cache_dir='/root/autodl-tmp/models'
)

# 然后用本地路径加载
from transformers import CLIPModel
model = CLIPModel.from_pretrained(model_dir)
```

#### 方案 6：手动下载模型

使用镜像站下载：

```bash
# 使用镜像站下载
HF_ENDPOINT=https://hf-mirror.com huggingface-cli download openai/clip-vit-base-patch32 --local-dir ./models/clip-vit-base-patch32
```

然后在代码中指定本地路径。

#### 方案 7：AutoDL 学术加速

AutoDL 提供学术加速服务，在终端运行：

```bash
# 开启学术加速（如果有这个服务）
source /etc/network_turbo  # 部分实例支持
```

或使用平台提供的加速脚本（在 AutoDL 控制台查看）。

### 快速验证网络连接

```bash
# 测试是否能访问 Hugging Face
curl -I https://huggingface.co

# 测试镜像站是否可用
curl -I https://hf-mirror.com
```

### 建议操作步骤

```bash
# 1. 先检查现有环境
echo $HF_ENDPOINT
ls /root/.cache/huggingface/hub/ 2>/dev/null | head -20

# 2. 设置镜像
export HF_ENDPOINT=https://hf-mirror.com

# 3. 重新运行程序
python -m code.main --build
```

---

## 其他常见问题

### Q1: FAISS 索引文件损坏

**症状**：加载索引时报错 `Cannot load index`

**解决方案**：
```bash
# 删除损坏的索引文件
rm -rf user_data/indexes/*

# 重新构建索引
python -m code.main --build
```

### Q2: OCR 识别失败

**症状**：图像处理时报 Tesseract 相关错误

**解决方案**：
```bash
# 检查 Tesseract 是否安装
tesseract --version

# 安装中文语言包（如缺失）
sudo apt-get install tesseract-ocr-chi-sim
```

### Q3: 内存不足

**症状**：处理大量图像时 OOM

**解决方案**：
- 减少批处理大小
- 使用更小的 CLIP 模型（如 `ViT-B/16`）
- 分批处理图像

---

*文档更新时间: 2026-02-13*
