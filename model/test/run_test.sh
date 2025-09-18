#!/bin/bash
# LLVIP测试运行脚本

# 设置脚本所在目录为工作目录
cd "$(dirname "$0")"

echo "测试脚本"
echo "================================"

# 检查Python环境
if ! command -v python &> /dev/null; then
    echo "错误: 未找到Python环境"
    exit 1
fi

# 默认配置文件
CONFIG_FILE="test_llvip_config.yaml"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --help|-h)
            echo "使用方法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --config FILE    指定配置文件 (默认: test_llvip_config.yaml)"
            echo "  --help, -h       显示此帮助信息"
            echo ""
            echo "示例:"
            echo "  $0                          # 使用默认配置"
            echo "  $0 --config my_config.yaml # 使用自定义配置"
            echo ""
            echo "您也可以直接运行 Python 脚本并传递更多参数:"
            echo "  python test_LLVIP.py --config test_llvip_config.yaml --batch_size 8"
            exit 0
            ;;
        *)
            echo "未知选项: $1"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件 '$CONFIG_FILE' 不存在"
    echo "请确保配置文件存在或使用 --config 指定正确的配置文件"
    exit 1
fi

echo "使用配置文件: $CONFIG_FILE"
echo "开始测试..."
echo ""

# 运行测试
CUDA_VISIBLE_DEVICES=0 python test.py --config "$CONFIG_FILE"
#python test.py --config "$CONFIG_FILE"
# 检查执行结果
if [ $? -eq 0 ]; then
    echo ""
    echo "测试完成!"
else
    echo ""
    echo "测试失败，请检查错误信息"
    exit 1
fi
