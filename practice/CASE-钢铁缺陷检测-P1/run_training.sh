#!/bin/bash
# 钢铁缺陷检测 - YOLOv11训练启动脚本

set -e  # 遇到错误时退出

echo "=========================================="
echo "钢铁缺陷检测 - YOLOv11训练"
echo "=========================================="

# 使用项目虚拟环境中的Python
PYTHON_EXEC=".venv/bin/python"

# 检查虚拟环境
if [ ! -f "$PYTHON_EXEC" ]; then
    echo "错误: 虚拟环境未找到，请先运行: uv sync"
    echo "或运行: uv venv && uv pip install ultralytics opencv-python pillow matplotlib seaborn tqdm pandas numpy scikit-learn pyyaml"
    exit 1
fi

# 检查依赖
echo "检查Python依赖..."
$PYTHON_EXEC -c "import ultralytics; print(f'Ultralytics版本: {ultralytics.__version__}')" || {
    echo "错误: 未安装ultralytics"
    echo "请运行: uv pip install --python .venv/bin/python ultralytics"
    exit 1
}

# 检查数据集
echo "检查数据集..."
if [ ! -f "data/yolo_format/data.yaml" ]; then
    echo "数据集未准备，开始转换..."
    $PYTHON_EXEC code/convert_voc_to_yolo.py --data-root data --val-ratio 0.2 --verify
fi

# 创建输出目录
mkdir -p runs/detect

# 训练模型
echo "开始训练YOLOv11模型..."
echo "使用配置: config/training_config.yaml"
echo "数据集: data/yolo_format/data.yaml"
echo "输出目录: runs/detect"
echo "Python解释器: $PYTHON_EXEC"
echo "=========================================="

$PYTHON_EXEC code/train_yolov11.py --config config/training_config.yaml

echo "=========================================="
echo "训练完成!"
echo "=========================================="

# 显示训练结果
echo "训练结果保存在:"
find runs/detect -name "best.pt" -type f | head -5

echo ""
echo "要重新训练或使用不同配置，可以运行:"
echo "  $PYTHON_EXEC code/train_yolov11.py --config config/training_config.yaml"
echo ""
echo "要仅分析数据集:"
echo "  $PYTHON_EXEC code/train_yolov11.py --config config/training_config.yaml --analyze-only"
echo ""
echo "要评估训练好的模型:"
echo "  $PYTHON_EXEC code/train_yolov11.py --config config/training_config.yaml --evaluate-only"
echo ""
echo "要导出模型为ONNX格式:"
echo "  $PYTHON_EXEC code/train_yolov11.py --config config/training_config.yaml --export-only --export-format onnx"