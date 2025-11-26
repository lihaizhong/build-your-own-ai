# 激活函数示例 - 快速参考

## 🚀 快速运行

### 激活函数可视化
```bash
cd code
uv run python3 plot_activation_functions.py    # 详细版（4子图）
uv run python3 simple_activation_plot.py      # 简化版（对比图）
```

### 激活函数导数分析
```bash
cd code
uv run python3 activation_derivatives.py      # 详细版（函数+导数）
uv run python3 simple_derivative_comparison.py # 简化版（导数对比）
```

## 📊 生成的文件

### 可视化结果 (`user_data/`)
- `activation_functions.png` - 详细激活函数图（4子图）
- `activation_functions_comparison.png` - 激活函数对比图
- `activation_derivatives.png` - 导数详细分析图
- `simple_derivative_comparison.png` - 导数对比图

## 🧮 关键数学公式

### 激活函数
- **Sigmoid**: σ(x) = 1/(1+e^(-x))
- **Tanh**: tanh(x) = (e^x-e^(-x))/(e^x+e^(-x))
- **ReLU**: ReLU(x) = max(0,x)

### 导数公式
- **Sigmoid**: σ'(x) = σ(x)(1-σ(x))
- **Tanh**: tanh'(x) = 1-tanh²(x)
- **ReLU**: ReLU'(x) = {1, x>0; 0, x≤0}

## 📈 反向传播特性

### 梯度消失严重程度
1. **Sigmoid** - 最严重（最大值0.25）
2. **Tanh** - 中等（最大值1.0）
3. **ReLU** - 最轻（缓解梯度消失）

### 实用建议
- **隐藏层**: 优先使用ReLU系列
- **输出层**: 二分类用Sigmoid
- **深层网络**: 避免Sigmoid和Tanh
- **初始化**: ReLU用He初始化

## 🎯 学习重点

1. **理解梯度消失原理** - 通过导数可视化
2. **掌握激活函数选择** - 根据任务需求
3. **认识反向传播影响** - 梯度对训练的影响
4. **应用最佳实践** - 实际项目中的选择策略