#!/usr/bin/env python3
import os
import sys

print("开始测试...")
print(f"Python 版本: {sys.version}")
print(f"当前工作目录: {os.getcwd()}")

try:
    print("导入 modelscope...")
    from modelscope import AutoModelForCausalLM, AutoTokenizer
    print("✅ modelscope 导入成功！")
except Exception as e:
    print(f"❌ modelscope 导入失败: {e}")
    sys.exit(1)

try:
    print("导入 torch...")
    import torch
    print(f"✅ torch 导入成功！版本: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 设备数量: {torch.cuda.device_count()}")
except Exception as e:
    print(f"❌ torch 导入失败: {e}")

# 检查模型路径
model_name = "./models/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
print(f"\n检查模型路径: {model_name}")
print(f"路径存在: {os.path.exists(model_name)}")

if os.path.exists(model_name):
    print("模型目录内容:")
    for item in os.listdir(model_name):
        print(f"  - {item}")

print("\n测试完成！")
