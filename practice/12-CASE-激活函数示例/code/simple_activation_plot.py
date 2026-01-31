import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def sigmoid(x):
    """Sigmoid激活函数"""
    return 1 / (1 + np.exp(-x))

def tanh_func(x):
    """Tanh激活函数"""
    return np.tanh(x)

def relu(x):
    """ReLU激活函数"""
    return np.maximum(0, x)

# 生成x轴数据
x = np.linspace(-5, 5, 1000)

# 计算三个激活函数的值
sigmoid_y = sigmoid(x)
tanh_y = tanh_func(x)
relu_y = relu(x)

# 创建图形
plt.figure(figsize=(10, 6))

# 绘制三个激活函数
plt.plot(x, sigmoid_y, 'b-', linewidth=2.5, label='Sigmoid σ(x) = 1/(1+e^(-x))', alpha=0.8)
plt.plot(x, tanh_y, 'r-', linewidth=2.5, label='Tanh tanh(x) = (e^x-e^(-x))/(e^x+e^(-x))', alpha=0.8)
plt.plot(x, relu_y, 'g-', linewidth=2.5, label='ReLU ReLU(x) = max(0,x)', alpha=0.8)

# 添加垂直线
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.8)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.3, linewidth=0.8)

# 设置坐标轴范围
plt.xlim(-5, 5)
plt.ylim(-1.5, 5.5)

# 添加网格
plt.grid(True, alpha=0.3, linestyle='--')

# 设置标题和标签
plt.title('常见激活函数对比图', fontsize=16, fontweight='bold')
plt.xlabel('输入值 (x)', fontsize=14)
plt.ylabel('输出值 f(x)', fontsize=14)

# 添加图例
plt.legend(fontsize=12, loc='upper left')

# 添加文本说明
plt.text(-4, 4, 'Sigmoid: (0,1)', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
plt.text(-4, 2.5, 'Tanh: (-1,1)', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
plt.text(1, 0.5, 'ReLU: [0,+∞)', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))

# 保存图像
plt.savefig('../user_data/activation_functions_comparison.png', dpi=300, bbox_inches='tight')
print("激活函数对比图已保存为 user_data/activation_functions_comparison.png")

# 显示图像
plt.show()