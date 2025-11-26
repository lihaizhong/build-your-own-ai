import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
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

# 计算激活函数及其导数
sigmoid_y = sigmoid(x)
sigmoid_deriv = sigmoid_derivative(x)

tanh_y = tanh_func(x)
tanh_deriv = tanh_derivative(x)

relu_y = relu(x)
relu_deriv = relu_derivative(x)

# 创建子图
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Sigmoid函数及其导数
axes[0, 0].plot(x, sigmoid_y, 'b-', linewidth=2.5, label='Sigmoid σ(x)', alpha=0.8)
axes[0, 0].plot(x, sigmoid_deriv, 'b--', linewidth=2.5, label="Sigmoid导数 σ'(x)", alpha=0.8)
axes[0, 0].set_title('Sigmoid函数及其导数', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('输入值 x')
axes[0, 0].set_ylabel('输出值')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend()
axes[0, 0].axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.8)
axes[0, 0].axvline(x=0, color='k', linestyle='-', alpha=0.3, linewidth=0.8)

# Tanh函数及其导数
axes[0, 1].plot(x, tanh_y, 'r-', linewidth=2.5, label='Tanh tanh(x)', alpha=0.8)
axes[0, 1].plot(x, tanh_deriv, 'r--', linewidth=2.5, label="Tanh导数 tanh'(x)", alpha=0.8)
axes[0, 1].set_title('Tanh函数及其导数', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('输入值 x')
axes[0, 1].set_ylabel('输出值')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend()
axes[0, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.8)
axes[0, 1].axvline(x=0, color='k', linestyle='-', alpha=0.3, linewidth=0.8)

# ReLU函数及其导数
axes[1, 0].plot(x, relu_y, 'g-', linewidth=2.5, label='ReLU ReLU(x)', alpha=0.8)
axes[1, 0].plot(x, relu_deriv, 'g--', linewidth=2.5, label="ReLU导数 ReLU'(x)", alpha=0.8)
axes[1, 0].set_title('ReLU函数及其导数', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('输入值 x')
axes[1, 0].set_ylabel('输出值')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend()
axes[1, 0].axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.8)
axes[1, 0].axvline(x=0, color='k', linestyle='-', alpha=0.3, linewidth=0.8)
axes[1, 0].set_ylim(-0.5, 1.5)

# 三个激活函数导数对比
axes[1, 1].plot(x, sigmoid_deriv, 'b-', linewidth=2.5, label='Sigmoid导数', alpha=0.8)
axes[1, 1].plot(x, tanh_deriv, 'r-', linewidth=2.5, label='Tanh导数', alpha=0.8)
axes[1, 1].plot(x, relu_deriv, 'g-', linewidth=2.5, label='ReLU导数', alpha=0.8)
axes[1, 1].set_title('激活函数导数对比', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('输入值 x')
axes[1, 1].set_ylabel('导数值')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].legend()
axes[1, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3, linewidth=0.8)
axes[1, 1].axvline(x=0, color='k', linestyle='-', alpha=0.3, linewidth=0.8)
axes[1, 1].set_ylim(-0.1, 1.1)

# 调整布局
plt.tight_layout()

# 保存图像
plt.savefig('../user_data/activation_derivatives.png', dpi=300, bbox_inches='tight')
print("激活函数导数图像已保存为 user_data/activation_derivatives.png")

# 显示图像
plt.show()

# 分析导数特性
print("\n=== 激活函数导数特性分析 ===")

print("\nSigmoid导数特性:")
print(f"  - 最大值: {np.max(sigmoid_deriv):.4f} (在x=0处)")
print(f"  - 导数范围: [{np.min(sigmoid_deriv):.4f}, {np.max(sigmoid_deriv):.4f}]")
print(f"  - 梯度消失问题: 严重，|x|>5时导数接近0")

print("\nTanh导数特性:")
print(f"  - 最大值: {np.max(tanh_deriv):.4f} (在x=0处)")
print(f"  - 导数范围: [{np.min(tanh_deriv):.4f}, {np.max(tanh_deriv):.4f}]")
print(f"  - 梯度消失问题: 比Sigmoid稍好，但|x|>3时仍然明显")

print("\nReLU导数特性:")
print(f"  - 值分布: {np.sum(relu_deriv == 0)} 个点为0，{np.sum(relu_deriv == 1)} 个点为1")
print(f"  - 激活比例: {np.sum(relu_deriv == 1) / len(relu_deriv) * 100:.1f}%")
print(f"  - 梯度消失问题: 死亡ReLU问题，但缓解了梯度消失")

print("\n=== 反向传播能力评估 ===")
print("梯度消失严重程度排序: Sigmoid > Tanh > ReLU")
print("计算复杂度排序: Tanh > Sigmoid > ReLU") 
print("实用性排名: ReLU > Tanh > Sigmoid")