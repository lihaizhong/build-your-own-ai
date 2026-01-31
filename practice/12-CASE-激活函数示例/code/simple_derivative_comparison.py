import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def sigmoid(x):
    """Sigmoid激活函数"""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Sigmoid函数的导数: σ'(x) = σ(x) * (1 - σ(x))"""
    s = sigmoid(x)
    return s * (1 - s)

def tanh_func(x):
    """Tanh激活函数"""
    return np.tanh(x)

def tanh_derivative(x):
    """Tanh函数的导数: tanh'(x) = 1 - tanh(x)²"""
    t = tanh_func(x)
    return 1 - t**2

def relu(x):
    """ReLU激活函数"""
    return np.maximum(0, x)

def relu_derivative(x):
    """ReLU函数的导数: ReLU'(x) = 1 if x > 0, else 0"""
    return np.where(x > 0, 1.0, 0.0)

# 生成x轴数据
x = np.linspace(-5, 5, 1000)

# 计算三个激活函数的导数
sigmoid_deriv = sigmoid_derivative(x)
tanh_deriv = tanh_derivative(x)
relu_deriv = relu_derivative(x)

# 创建图形
plt.figure(figsize=(12, 8))

# 绘制三个激活函数的导数
plt.plot(x, sigmoid_deriv, 'b-', linewidth=3, label="Sigmoid导数 σ'(x) = σ(x)(1-σ(x))", alpha=0.9)
plt.plot(x, tanh_deriv, 'r-', linewidth=3, label="Tanh导数 tanh'(x) = 1-tanh²(x)", alpha=0.9)
plt.plot(x, relu_deriv, 'g-', linewidth=3, label="ReLU导数 ReLU'(x) = {1,x>0; 0,x≤0}", alpha=0.9)

# 添加辅助线
plt.axhline(y=0, color='k', linestyle='-', alpha=0.4, linewidth=1)
plt.axvline(x=0, color='k', linestyle='-', alpha=0.4, linewidth=1)

# 设置坐标轴范围
plt.xlim(-5, 5)
plt.ylim(-0.1, 1.1)

# 添加网格
plt.grid(True, alpha=0.3, linestyle='--')

# 设置标题和标签
plt.title('激活函数导数对比图 - 反向传播梯度分析', fontsize=16, fontweight='bold')
plt.xlabel('输入值 (x)', fontsize=14)
plt.ylabel('导数值 f\'(x)', fontsize=14)

# 添加图例
plt.legend(fontsize=12, loc='upper right')

# 添加问题标注
plt.text(-4, 0.8, '梯度消失\n严重', fontsize=11, bbox=dict(boxstyle="round,pad=0.4", facecolor="lightblue", alpha=0.7))
plt.text(-1, 0.6, '梯度消失\n中等', fontsize=11, bbox=dict(boxstyle="round,pad=0.4", facecolor="lightcoral", alpha=0.7))
plt.text(2, 0.8, '缓解梯度消失\n但有死亡ReLU', fontsize=11, bbox=dict(boxstyle="round,pad=0.4", facecolor="lightgreen", alpha=0.7))

# 添加极值点标注
plt.scatter([0], [0.25], s=100, c='blue', alpha=0.7, zorder=5)
plt.scatter([0], [1], s=100, c='red', alpha=0.7, zorder=5)
plt.text(0.2, 0.25, 'σ\'(0)=0.25', fontsize=10, color='blue')
plt.text(0.2, 1, 'tanh\'(0)=1', fontsize=10, color='red')

# 保存图像
plt.savefig('../user_data/simple_derivative_comparison.png', dpi=300, bbox_inches='tight')
print("激活函数导数对比图已保存为 user_data/simple_derivative_comparison.png")

# 显示图像
plt.show()

# 详细分析导数特性
print("\n=== 详细导数特性分析 ===")

# Sigmoid导数分析
sigmoid_max = np.max(sigmoid_deriv)
sigmoid_at_2 = sigmoid_derivative(2)
sigmoid_at_5 = sigmoid_derivative(5)
print(f"\nSigmoid导数 σ'(x):")
print(f"  - 数学公式: σ'(x) = σ(x)(1-σ(x))")
print(f"  - 最大值: {sigmoid_max:.4f} (在x=0处)")
print(f"  - 在x=2处: {sigmoid_at_2:.4f}")
print(f"  - 在x=5处: {sigmoid_at_5:.4f}")
print(f"  - 问题: 梯度消失严重，深层网络训练困难")

# Tanh导数分析
tanh_max = np.max(tanh_deriv)
tanh_at_2 = tanh_derivative(2)
tanh_at_3 = tanh_derivative(3)
print(f"\nTanh导数 tanh'(x):")
print(f"  - 数学公式: tanh'(x) = 1 - tanh²(x)")
print(f"  - 最大值: {tanh_max:.4f} (在x=0处)")
print(f"  - 在x=2处: {tanh_at_2:.4f}")
print(f"  - 在x=3处: {tanh_at_3:.4f}")
print(f"  - 改进: 比Sigmoid收敛快，但仍有梯度消失")

# ReLU导数分析
relu_positive_ratio = np.sum(relu_deriv == 1) / len(relu_deriv)
print(f"\nReLU导数 ReLU'(x):")
print(f"  - 数学公式: ReLU'(x) = {{1, x>0; 0, x≤0}}")
print(f"  - 正值比例: {relu_positive_ratio*100:.1f}%")
print(f"  - 优势: 缓解梯度消失，计算简单")
print(f"  - 问题: 死亡ReLU，50%神经元可能不激活")

print(f"\n=== 反向传播实用建议 ===")
print("1. 隐藏层: 推荐使用ReLU及其变体(Leaky ReLU, ELU等)")
print("2. 输出层: 二分类用Sigmoid，多分类用Softmax")
print("3. 深层网络: 避免使用Sigmoid和Tanh")
print("4. 初始化: ReLU网络使用He初始化，Sigmoid/Tanh使用Xavier初始化")
print("5. 批归一化: 可配合ReLU使用，进一步缓解梯度问题")