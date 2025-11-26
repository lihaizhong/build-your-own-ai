import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as path_effects

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
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

# 创建一个具体的示例：神经网络中的激活过程
# 假设我们有4个输入值，经过线性变换后得到不同的输入值
input_values = np.array([-2.5, -0.5, 0.8, 2.0, 3.5])
input_labels = ['神经元A', '神经元B', '神经元C', '神经元D', '神经元E']

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

# 创建图形 - 使用更大的画布来容纳所有内容
fig = plt.figure(figsize=(20, 14))

# 创建网格布局 - 修复布局问题
gs = fig.add_gridspec(4, 3, height_ratios=[1.5, 1.5, 1.5, 1], 
                      hspace=0.35, wspace=0.25)

# 第一行：激活函数单独展示
ax_sigmoid = fig.add_subplot(gs[0, 0])
ax_tanh = fig.add_subplot(gs[0, 1])
ax_relu = fig.add_subplot(gs[0, 2])

# 第二行：激活函数对比和总结
ax_combined = fig.add_subplot(gs[1, :2])
ax_stats = fig.add_subplot(gs[1, 2])

# 单独展示Sigmoid函数
ax_sigmoid.plot(x_smooth, sigmoid_y, 'b-', linewidth=3, label='Sigmoid σ(x)', alpha=0.8)
ax_sigmoid.scatter(input_values, sigmoid_output, s=100, c='blue', marker='o', 
           alpha=0.7, edgecolor='navy', linewidth=2, zorder=5)

# 只标注部分重要点，避免重叠
important_indices = [0, 2, 4]  # A, C, E
for i in important_indices:
    x, y = input_values[i], sigmoid_output[i]
    ax_sigmoid.annotate(f'{input_labels[i]}\n({x:.1f} → {y:.2f})', 
                (x, y), xytext=(5, 5), textcoords='offset points',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
                fontsize=9, ha='center')

ax_sigmoid.set_title('Sigmoid激活函数\n输出范围: (0, 1)', fontsize=13, fontweight='bold')
ax_sigmoid.set_xlabel('输入值 x', fontsize=11)
ax_sigmoid.set_ylabel('输出值 f(x)', fontsize=11)
ax_sigmoid.grid(True, alpha=0.3)
ax_sigmoid.legend()
ax_sigmoid.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax_sigmoid.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)

# 单独展示Tanh函数
ax_tanh.plot(x_smooth, tanh_y, 'r-', linewidth=3, label='Tanh tanh(x)', alpha=0.8)
ax_tanh.scatter(input_values, tanh_output, s=100, c='red', marker='s', 
           alpha=0.7, edgecolor='darkred', linewidth=2, zorder=5)

# 只标注重要点
for i in important_indices:
    x, y = input_values[i], tanh_output[i]
    ax_tanh.annotate(f'{input_labels[i]}\n({x:.1f} → {y:.2f})', 
                (x, y), xytext=(5, 5), textcoords='offset points',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.8),
                fontsize=9, ha='center')

ax_tanh.set_title('Tanh激活函数\n输出范围: (-1, 1)', fontsize=13, fontweight='bold')
ax_tanh.set_xlabel('输入值 x', fontsize=11)
ax_tanh.set_ylabel('输出值 f(x)', fontsize=11)
ax_tanh.grid(True, alpha=0.3)
ax_tanh.legend()
ax_tanh.axhline(y=0, color='k', linestyle='-', alpha=0.3)

# 单独展示ReLU函数
ax_relu.plot(x_smooth, relu_y, 'g-', linewidth=3, label='ReLU(x)', alpha=0.8)
ax_relu.scatter(input_values, relu_output, s=100, c='green', marker='^', 
               alpha=0.7, edgecolor='darkgreen', linewidth=2, zorder=5)

# 只标注重要点
for i in important_indices:
    x, y = input_values[i], relu_output[i]
    ax_relu.annotate(f'{input_labels[i]}\n({x:.1f} → {y:.1f})', 
                    (x, y), xytext=(5, 5), textcoords='offset points',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.8),
                    fontsize=9, ha='center')

ax_relu.set_title('ReLU激活函数\n输出范围: [0, +∞)', fontsize=13, fontweight='bold')
ax_relu.set_xlabel('输入值 x', fontsize=11)
ax_relu.set_ylabel('输出值 f(x)', fontsize=11)
ax_relu.grid(True, alpha=0.3)
ax_relu.legend()
ax_relu.axhline(y=0, color='k', linestyle='-', alpha=0.3)

# 三个函数对比图
ax_combined.plot(x_smooth, sigmoid_y, 'b-', linewidth=2.5, label='Sigmoid', alpha=0.8)
ax_combined.plot(x_smooth, tanh_y, 'r-', linewidth=2.5, label='Tanh', alpha=0.8)
ax_combined.plot(x_smooth, relu_y, 'g-', linewidth=2.5, label='ReLU', alpha=0.8)

# 添加标注点（只标注重要的）
for i in important_indices:
    x = input_values[i]
    ax_combined.scatter(x, sigmoid_output[i], s=50, c='blue', marker='o', alpha=0.6)
    ax_combined.scatter(x, tanh_output[i], s=50, c='red', marker='s', alpha=0.6)
    ax_combined.scatter(x, relu_output[i], s=50, c='green', marker='^', alpha=0.6)

ax_combined.set_title('激活函数对比图', fontsize=14, fontweight='bold')
ax_combined.set_xlabel('输入值 x', fontsize=12)
ax_combined.set_ylabel('输出值 f(x)', fontsize=12)
ax_combined.grid(True, alpha=0.3)
ax_combined.legend()
ax_combined.axhline(y=0, color='k', linestyle='-', alpha=0.3)

# 第三行：数值对比表格
ax_table = fig.add_subplot(gs[2, :])
ax_table.axis('off')

# 创建对比表格
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
headers = ['神经元', '输入值', 'Sigmoid输出', 'Tanh输出', 'ReLU输出']
table_data.insert(0, headers)

# 创建表格 - 使用更紧凑的布局
table = ax_table.table(cellText=table_data[1:], colLabels=table_data[0],
                      cellLoc='center', loc='center',
                      colWidths=[0.15, 0.15, 0.175, 0.175, 0.175])

# 设置表格样式
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 1.8)

# 设置表头样式
for i in range(len(headers)):
    table[(0, i)].set_facecolor('#4CAF50')
    table[(0, i)].set_text_props(weight='bold', color='white', fontsize=10)

# 设置数据行样式
colors = ['#E3F2FD', '#FFF3E0', '#E8F5E8', '#F3E5F5', '#E0F2F1']
for i in range(1, len(table_data)):
    color = colors[(i-1) % len(colors)]
    for j in range(len(headers)):
        table[(i, j)].set_facecolor(color)
        table[(i, j)].set_text_props(fontsize=10)

ax_table.set_title('激活函数数值对比表 - 相同输入的不同输出', fontsize=15, fontweight='bold', pad=15)

# 第三行右侧：特性统计
ax_stats.axis('off')
stats_text = f"""
📊 输出范围分析:
Sigmoid: [{sigmoid_output.min():.3f}, {sigmoid_output.max():.3f}]
Tanh: [{tanh_output.min():.3f}, {tanh_output.max():.3f}]
ReLU: [{relu_output.min():.1f}, {relu_output.max():.1f}]

🎯 梯度特性:
Sigmoid: 梯度最大值 ≈ 0.25
Tanh: 梯度最大值 ≈ 1.0
ReLU: 梯度为1或0

⚡ 计算复杂度:
Sigmoid: 指数运算 - 较慢
Tanh: 指数运算 - 较慢  
ReLU: 简单比较 - 最快
"""
ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
              fontsize=10, verticalalignment='top', fontfamily='monospace',
              bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5f5f5", alpha=0.9))

# 第四行：特性分析和应用场景
ax_analysis = fig.add_subplot(gs[3, :])
ax_analysis.axis('off')

# 创建特性分析文本
analysis_text = """
🔵 Sigmoid函数特性: 输出范围(0,1) • 平滑连续 • 梯度消失严重 • 适合输出层二分类     🔴 Tanh函数特性: 输出范围(-1,1) • 以0为中心 • 收敛较快 • 适合RNN等循环网络     🟢 ReLU函数特性: 输出范围[0,+∞) • 计算高效 • 缓解梯度消失 • 现代深度学习首选
"""

# 添加特性分析文本
ax_analysis.text(0.5, 0.5, analysis_text, transform=ax_analysis.transAxes,
                fontsize=12, verticalalignment='center', horizontalalignment='center',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", alpha=0.9))

# 设置整体标题
fig.suptitle('神经网络激活函数深度对比分析\n基于5个神经元实例的详细分析', 
            fontsize=20, fontweight='bold', y=0.98)

# 添加说明文本
fig.text(0.5, 0.02, 
         '💡 关键洞察: 相同的输入经过不同激活函数产生不同输出，体现了激活函数在神经网络中的核心作用\n'
         '选择激活函数需要考虑输出范围、梯度特性、计算效率和具体应用场景',
         ha='center', fontsize=12, style='italic',
         bbox=dict(boxstyle="round,pad=0.5", facecolor="#fffacd", alpha=0.8))

# 保存图像
plt.savefig('../user_data/activation_comparison_detailed.png', dpi=300, bbox_inches='tight')
print("详细激活函数对比图已保存为 user_data/activation_comparison_detailed.png")

# 显示图像
plt.show()

# 输出详细分析
print("\n=== 激活函数实际应用分析 ===")
print("基于5个神经元输入值的对比:")
print(f"{'神经元':<8} {'输入值':<8} {'Sigmoid':<10} {'Tanh':<10} {'ReLU':<10} {'最佳选择'}")
print("-" * 65)

for i, label in enumerate(input_labels):
    x = input_values[i]
    sig = sigmoid_output[i]
    tan = tanh_output[i]
    rel = relu_output[i]
    
    # 判断最佳选择
    if x < -1:
        best = "ReLU (负值变0)"
    elif -1 <= x <= 1:
        best = "Tanh (适中输出)"
    else:
        best = "ReLU (保持线性)"
    
    print(f"{label:<8} {x:<8.1f} {sig:<10.3f} {tan:<10.3f} {rel:<10.1f} {best}")

print(f"\n=== 关键发现 ===")
print(f"1. Sigmoid输出范围稳定在(0,1)，适合概率解释")
print(f"2. Tanh以0为中心对称，负值输入产生负值输出")
print(f"3. ReLU对负值直接截断为0，计算效率最高")
print(f"4. 深度网络应优先选择ReLU，输出层根据任务选择Sigmoid/Softmax")