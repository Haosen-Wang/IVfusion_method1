#!/bin/bash

# 像素值分析脚本 - 快速分析图像中的亮区域

IMAGE_PATH="/data/1024whs_data/DeMMI-RF/Train/infrared/noise50/LLVIP/010066.jpg"
SAVE_DIR="./pixel_analysis"

echo "=========================================="
echo "像素值分析 - 红外图像"
echo "=========================================="

# 1. 基础分析 (P90 阈值)
echo -e "\n[1] 使用 P90 阈值分析..."
python analyze_pixel_values.py \
    --image "$IMAGE_PATH" \
    --mode L \
    --percentile 90 \
    --save_dir "$SAVE_DIR/p90" \
    --compare

# 2. 更严格的分析 (P95 阈值，只移除最亮的 5%)
echo -e "\n[2] 使用 P95 阈值分析..."
python analyze_pixel_values.py \
    --image "$IMAGE_PATH" \
    --mode L \
    --percentile 95 \
    --save_dir "$SAVE_DIR/p95" \
    --compare

# 3. 只移除极亮区域 (P98 阈值)
echo -e "\n[3] 使用 P98 阈值分析..."
python analyze_pixel_values.py \
    --image "$IMAGE_PATH" \
    --mode L \
    --percentile 98 \
    --save_dir "$SAVE_DIR/p98" \
    --compare

# 4. 使用固定阈值 (0.7, 约 178/255)
echo -e "\n[4] 使用固定阈值 0.7..."
python analyze_pixel_values.py \
    --image "$IMAGE_PATH" \
    --mode L \
    --threshold 0.7 \
    --save_dir "$SAVE_DIR/threshold_0.7" \
    --compare

echo -e "\n=========================================="
echo "分析完成！结果保存在: $SAVE_DIR"
echo "=========================================="
