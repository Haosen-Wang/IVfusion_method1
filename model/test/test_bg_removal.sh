#!/bin/bash

# 红外背景移除功能测试脚本
# 对比不同阈值和方法的效果

MODEL="/data/1024whs_checkpoint/CIV/Train_DIV_LLVIP_visible/latest_checkpoint.pth"
I_IMG="/data/1024whs_data/DeMMI-RF/Train/infrared/noise50/LLVIP/010066.jpg"
V_IMG="/data/1024whs_data/DeMMI-RF/Train/visible/noise50/LLVIP/010066.jpg"
D_IMG="/data/1024whs_data/DeMMI-RF/Train/degrad/noise50/LLVIP/010066.jpg"
TASK="dv_i"
DEVICE="cuda:0"
GPU=2

echo "========================================"
echo "红外背景移除功能测试"
echo "========================================"

# 测试1: P95 + subtract (推荐配置)
echo -e "\n[1] 测试 P95 + subtract (推荐)"
CUDA_VISIBLE_DEVICES=$GPU python test_single_visualize.py \
    --model "$MODEL" \
    --i_img "$I_IMG" \
    --v_img "$V_IMG" \
    --d_img "$D_IMG" \
    --task "$TASK" \
    --device "$DEVICE" \
    --save_dir ./output/test_p95_subtract \
    --remove_ir_bg \
    --bg_percentile 95 \
    --bg_method subtract

# 测试2: P98 + subtract (保守配置)
echo -e "\n[2] 测试 P98 + subtract (保守)"
CUDA_VISIBLE_DEVICES=$GPU python test_single_visualize.py \
    --model "$MODEL" \
    --i_img "$I_IMG" \
    --v_img "$V_IMG" \
    --d_img "$D_IMG" \
    --task "$TASK" \
    --device "$DEVICE" \
    --save_dir ./output/test_p98_subtract \
    --remove_ir_bg \
    --bg_percentile 98 \
    --bg_method subtract

# 测试3: P95 + blend (温和处理)
echo -e "\n[3] 测试 P95 + blend (温和)"
CUDA_VISIBLE_DEVICES=$GPU python test_single_visualize.py \
    --model "$MODEL" \
    --i_img "$I_IMG" \
    --v_img "$V_IMG" \
    --d_img "$D_IMG" \
    --task "$TASK" \
    --device "$DEVICE" \
    --save_dir ./output/test_p95_blend \
    --remove_ir_bg \
    --bg_percentile 95 \
    --bg_method blend

# 测试4: P95 + mask (完全移除)
echo -e "\n[4] 测试 P95 + mask (完全移除)"
CUDA_VISIBLE_DEVICES=$GPU python test_single_visualize.py \
    --model "$MODEL" \
    --i_img "$I_IMG" \
    --v_img "$V_IMG" \
    --d_img "$D_IMG" \
    --task "$TASK" \
    --device "$DEVICE" \
    --save_dir ./output/test_p95_mask \
    --remove_ir_bg \
    --bg_percentile 95 \
    --bg_method mask

# 测试5: 不移除背景（对比基准）
echo -e "\n[5] 测试 不移除背景（对比基准）"
CUDA_VISIBLE_DEVICES=$GPU python test_single_visualize.py \
    --model "$MODEL" \
    --i_img "$I_IMG" \
    --v_img "$V_IMG" \
    --d_img "$D_IMG" \
    --task "$TASK" \
    --device "$DEVICE" \
    --save_dir ./output/test_no_removal \
    --no_remove_ir_bg

echo -e "\n========================================"
echo "所有测试完成！"
echo "结果保存在 ./output/ 目录"
echo "========================================"
echo -e "\n对比结果:"
echo "  test_p95_subtract/  - P95+subtract (推荐)"
echo "  test_p98_subtract/  - P98+subtract (保守)"
echo "  test_p95_blend/     - P95+blend (温和)"
echo "  test_p95_mask/      - P95+mask (完全移除)"
echo "  test_no_removal/    - 不移除背景（基准）"
