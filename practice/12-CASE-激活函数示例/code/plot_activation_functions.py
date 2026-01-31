import numpy as np
import matplotlib.pyplot as plt

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
plt.figure(figsize=(12, 8))

# 绘制三个激活函数
plt.subplot(2, 2, 1)
plt.plot(x, sigmoid_y, 'b-', linewidth=2, label='Sigmoid')
plt.title('Sigmoid Activation Function')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(x, tanh_y, 'r-', linewidth=2, label='Tanh')
plt.title('Tanh Activation Function')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(x, relu_y, 'g-', linewidth=2, label='ReLU')
plt.title('ReLU Activation Function')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True, alpha=0.3)
plt.legend()

# 在一个图中显示所有三个函数
plt.subplot(2, 2, 4)
plt.plot(x, sigmoid_y, 'b-', linewidth=2, label='Sigmoid', alpha=0.7)
plt.plot(x, tanh_y, 'r-', linewidth=2, label='Tanh', alpha=0.7)
plt.plot(x, relu_y, 'g-', linewidth=2, label='ReLU', alpha=0.7)
plt.title('Comparison of Activation Functions')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True, alpha=0.3)
plt.legend()

# 调整布局
plt.tight_layout()

# 保存图像
plt.savefig('../user_data/activation_functions.png', dpi=300, bbox_inches='tight')
print("激活函数图像已保存为 user_data/activation_functions.png")

# 显示图像
plt.show()

# 打印一些关键信息
print("\n=== 激活函数特性总结 ===")
print("Sigmoid函数:")
print("  - 输出范围: (0, 1)")
print("  - 特点: S型曲线，用于二分类问题")
print("  - 缺点: 梯度消失问题")

print("\nTanh函数:")
print("  - 输出范围: (-1, 1)")
print("  - 特点: 关于原点对称")
print("  - 缺点: 梯度消失问题")

print("\nReLU函数:")
print("  - 输出范围: [0, +∞)")
print("  - 特点: 计算简单，缓解梯度消失")
print("  - 缺点: 死亡ReLU问题")