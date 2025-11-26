# 激活函数示例 - 快速开始

## 📖 项目文档

本项目使用 **IFLOW.md** 作为主要文档，包含：

### 📋 完整内容
- **项目概述**: 功能介绍和技术架构
- **快速使用**: 运行指南和命令示例  
- **理论背景**: 激活函数数学原理
- **导数分析**: 反向传播特性详解
- **最佳实践**: 实际应用建议
- **扩展指南**: 功能扩展方向

## 🚀 立即开始

### 1. 查看完整文档
```bash
cat IFLOW.md
```

### 2. 运行示例代码
```bash
cd code
uv run python3 plot_activation_functions.py    # 激活函数可视化
uv run python3 activation_derivatives.py      # 导数分析
```

### 3. 查看快速参考
```bash
cat docs/QUICK_REFERENCE.md
```

## 📁 项目结构
```
CASE-激活函数示例/
├── IFLOW.md                   # 📚 完整项目文档（主要文档）
├── docs/                      # 辅助文档目录
│   ├── README.md              # 文档索引
│   ├── START.md               # 🚀 本文件 - 快速开始
│   └── QUICK_REFERENCE.md     # ⚡ 快速参考指南
├── code/                      # Python代码
└── user_data/                 # 生成的文件
```

## 💡 提示
- 第一次使用建议先阅读 **IFLOW.md** 的项目概述部分
- 想要快速操作参考 **docs/QUICK_REFERENCE.md**
- 遇到问题时查看 **IFLOW.md** 的故障排除部分
- 了解文档结构查看项目根目录结构