#!/bin/bash

# 单图测试脚本示例
# 使用方法: bash run_single_test.sh

# 设置路径
MODEL_PATH="path/to/your/model.pth"  # 修改为你的模型路径
I_IMAGE="path/to/infrared.png"        # 修改为红外图像路径
V_IMAGE="path/to/visible.png"         # 修改为可见光图像路径
D_IMAGE="path/to/degraded.png"        # 修改为退化图像路径
TASK="dv_i"                           # 任务类型: dv_i 或 di_v
DEVICE="cuda:0"                       # 设备: cuda:0 或 cpu
SAVE_DIR="./single_test_output"       # 保存目录

# 运行测试
python test_single_visualize.py \
    --model "$MODEL_PATH" \
    --i_img "$I_IMAGE" \
    --v_img "$V_IMAGE" \
    --d_img "$D_IMAGE" \
    --task "$TASK" \
    --device "$DEVICE" \
    --save_dir "$SAVE_DIR"

echo "测试完成！结果保存在 $SAVE_DIR"
