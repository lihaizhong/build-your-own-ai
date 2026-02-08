import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体和样式 - 优化字体设置
plt.rcParams['font.family'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

def sigmoid(x):
    """Sigmoid激活函数"""
    return 1 / (1 + np.exp(-x))

def tanh_func(x):
    """Tanh激活函数"""
    return np.tanh(x)

def relu(x):
    """ReLU激活函数"""
    return np.maximum(0, x)

# 创建一个具体的示例：神经网络中的激活过程
input_values = np.array([-2.5, -0.5, 0.8, 2.0, 3.5])
input_labels = ['Neuron A', 'Neuron B', 'Neuron C', 'Neuron D', 'Neuron E']

# 生成平滑的x轴数据用于绘制曲线
x_smooth = np.linspace(-5, 5, 1000)

# 计算激活函数值
sigmoid_y = sigmoid(x_smooth)
tanh_y = tanh_func(x_smooth)
relu_y = relu(x_smooth)

# 计算具体示例的激活值
sigmoid_output = sigmoid(input_values)
tanh_output = tanh_func(input_values)
relu_output = relu(input_values)

# 创建图形 - 优化的布局
fig = plt.figure(figsize=(22, 16))
gs = fig.add_gridspec(4, 4, height_ratios=[1, 1, 1, 0.8], 
                      hspace=0.4, wspace=0.3)

# 第一行：单独展示激活函数（避免重叠）
colors = {'sigmoid': '#1f77b4', 'tanh': '#d62728', 'relu': '#2ca02c'}
titles = {
    'sigmoid': 'Sigmoid Activation Function\nOutput Range: (0, 1)',
    'tanh': 'Tanh Activation Function\nOutput Range: (-1, 1)', 
    'relu': 'ReLU Activation Function\nOutput Range: [0, +∞)'
}

functions = {'sigmoid': sigmoid_y, 'tanh': tanh_y, 'relu': relu_y}
outputs = {'sigmoid': sigmoid_output, 'tanh': tanh_output, 'relu': relu_output}

for i, func_name in enumerate(['sigmoid', 'tanh', 'relu']):
    ax = fig.add_subplot(gs[0, i])
    func_y = functions[func_name]
    func_output = outputs[func_name]
    
    # 绘制激活函数曲线
    ax.plot(x_smooth, func_y, color=colors[func_name], linewidth=3, 
            label=func_name.title(), alpha=0.8)
    
    # 绘制示例点
    markers = {'sigmoid': 'o', 'tanh': 's', 'relu': '^'}
    ax.scatter(input_values, func_output, s=120, c=colors[func_name], 
              marker=markers[func_name], alpha=0.7, edgecolor='white', 
              linewidth=2, zorder=5)
    
    # 只标注关键点，避免重叠
    key_indices = [0, 2, 4]  # A, C, E
    for idx in key_indices:
        x, y = input_values[idx], func_output[idx]
        ax.annotate(f'{input_labels[idx]}\n({x:.1f} → {y:.2f})', 
                   (x, y), xytext=(8, 8), textcoords='offset points',
                   bbox=dict(boxstyle="round,pad=0.3", 
                           facecolor=colors[func_name], alpha=0.3),
                   fontsize=9, ha='center')
    
    ax.set_title(titles[func_name], fontsize=13, fontweight='bold', pad=10)
    ax.set_xlabel('Input Value (x)', fontsize=11)
    ax.set_ylabel('Output f(x)', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=10)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)

# 空白图表位置
fig.add_subplot(gs[0, 3]).axis('off')

# 第二行：对比图和特性分析
ax_combined = fig.add_subplot(gs[1, :2])
ax_combined.plot(x_smooth, sigmoid_y, color=colors['sigmoid'], linewidth=3, 
                label='Sigmoid', alpha=0.8)
ax_combined.plot(x_smooth, tanh_y, color=colors['tanh'], linewidth=3, 
                label='Tanh', alpha=0.8)
ax_combined.plot(x_smooth, relu_y, color=colors['relu'], linewidth=3, 
                label='ReLU', alpha=0.8)

# 添加关键点标注
for i, func_name in enumerate(['sigmoid', 'tanh', 'relu']):
    marker = markers[func_name]
    ax_combined.scatter(input_values, outputs[func_name], s=80, 
                      c=colors[func_name], marker=marker, alpha=0.6, 
                      edgecolor='white', linewidth=1.5)

ax_combined.set_title('Activation Functions Comparison', fontsize=16, fontweight='bold')
ax_combined.set_xlabel('Input Value (x)', fontsize=12)
ax_combined.set_ylabel('Output f(x)', fontsize=12)
ax_combined.grid(True, alpha=0.3)
ax_combined.legend(fontsize=11, loc='upper left')
ax_combined.axhline(y=0, color='k', linestyle='-', alpha=0.3)

# 特性分析图表
ax_properties = fig.add_subplot(gs[1, 2:])
ax_properties.axis('off')

# 创建特性分析文本
properties_text = f"""
ACTIVATION FUNCTION PROPERTIES

Sigmoid Function:
• Output Range: (0, 1) - Natural for probability
• Gradient: Maximum ≈ 0.25 (vanishing gradient issue)
• Computation: Exponential operations - slower
• Applications: Output layer for binary classification

Tanh Function:
• Output Range: (-1, 1) - Zero-centered
• Gradient: Maximum ≈ 1.0 (still vanishes)
• Computation: Exponential operations - slower
• Applications: Hidden layers, RNN networks

ReLU Function:
• Output Range: [0, +∞) - Efficient computation
• Gradient: 1 or 0 (no vanishing)
• Computation: Simple comparison - fastest
• Applications: Deep networks, modern DL standard
"""

ax_properties.text(0.05, 0.95, properties_text, transform=ax_properties.transAxes,
                  fontsize=10, verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle="round,pad=0.5", facecolor="#f8f9fa", alpha=0.9))

# 第三行：详细数值对比表格
ax_table = fig.add_subplot(gs[2, :])
ax_table.axis('off')

# 创建对比表格数据
table_data = []
for i, label in enumerate(input_labels):
    table_data.append([
        label,
        f'{input_values[i]:.1f}',
        f'{sigmoid_output[i]:.3f}',
        f'{tanh_output[i]:.3f}',
        f'{relu_output[i]:.3f}'
    ])

# 添加表头
headers = ['Neuron', 'Input', 'Sigmoid Output', 'Tanh Output', 'ReLU Output']
table_data.insert(0, headers)

# 创建表格
table = ax_table.table(cellText=table_data[1:], colLabels=table_data[0],
                      cellLoc='center', loc='center',
                      colWidths=[0.12, 0.12, 0.19, 0.19, 0.19])

# 设置表格样式
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2)

# 设置表头样式
for i in range(len(headers)):
    table[(0, i)].set_facecolor('#495057')
    table[(0, i)].set_text_props(weight='bold', color='white', fontsize=10)

# 设置数据行样式
row_colors = ['#e3f2fd', '#fff3e0', '#e8f5e8', '#f3e5f5', '#e0f2f1']
for i in range(1, len(table_data)):
    color = row_colors[(i-1) % len(row_colors)]
    for j in range(len(headers)):
        table[(i, j)].set_facecolor(color)
        table[(i, j)].set_text_props(fontsize=10)

ax_table.set_title('Activation Functions Numerical Comparison', 
                  fontsize=16, fontweight='bold', pad=20)

# 第四行：关键洞察
ax_insight = fig.add_subplot(gs[3, :])
ax_insight.axis('off')

insight_text = """
KEY INSIGHTS: Same input through different activation functions produces different outputs, demonstrating the core role of activation functions in neural networks.

The choice of activation function should consider: output range, gradient characteristics, computational efficiency, and specific application scenarios.

Best Practices: Hidden layers should prioritize ReLU; output layers should choose based on task (Sigmoid for binary, Linear for regression, Softmax for multi-class).
"""

ax_insight.text(0.5, 0.5, insight_text, transform=ax_insight.transAxes,
               fontsize=12, verticalalignment='center', horizontalalignment='center',
               bbox=dict(boxstyle="round,pad=0.8", facecolor="#fff3cd", 
                        edgecolor='#ffeaa7', alpha=0.9))

# 设置整体标题
fig.suptitle('Deep Learning Activation Functions Analysis\n'
            'Comprehensive Comparison Based on 5 Neuron Examples', 
            fontsize=20, fontweight='bold', y=0.97)

# 保存图像
plt.savefig('../user_data/activation_comparison_optimized.png', dpi=300, 
           bbox_inches='tight', facecolor='white')
print("优化后的激活函数对比图已保存为 user_data/activation_comparison_optimized.png")

# 关闭图表（不显示）
plt.close()

# 输出分析报告
print("\n" + "="*70)
print("ACTIVATION FUNCTIONS ANALYSIS REPORT")
print("="*70)
print(f"{'Neuron':<10} {'Input':<8} {'Sigmoid':<10} {'Tanh':<10} {'ReLU':<10} {'Best Choice'}")
print("-" * 70)

recommendations = {
    -2.5: "ReLU (Negative → 0)",
    -0.5: "Tanh (Moderate output)", 
    0.8: "Tanh (Balanced)",
    2.0: "ReLU (Linear preserve)",
    3.5: "ReLU (Linear preserve)"
}

for i, label in enumerate(input_labels):
    x = input_values[i]
    sig = sigmoid_output[i]
    tan = tanh_output[i]
    rel = relu_output[i]
    best = recommendations[x]
    
    print(f"{label:<10} {x:<8.1f} {sig:<10.3f} {tan:<10.3f} {rel:<10.1f} {best}")

print("\n" + "="*70)
print("CRITICAL FINDINGS:")
print("• Sigmoid: Stable output range (0,1) - perfect for probability")
print("• Tanh: Zero-centered symmetry - negative inputs produce negative outputs") 
print("• ReLU: Directly clips negatives to 0 - highest computational efficiency")
print("• Deep networks: Prioritize ReLU; output layer: task-dependent")
print("="*70)