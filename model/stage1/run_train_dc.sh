
#!/bin/bash

# =============================================================================
# 训练退化恢复模型脚本 (支持YAML配置)
# 使用方法: 
#   1. 使用YAML配置文件: ./run_train_dc.sh --config config.yaml
#   2. 使用YAML配置文件和指定实验: ./run_train_dc.sh --config config.yaml --experiment small_model
#   3. 使用YAML配置文件并指定GPU: ./run_train_dc.sh --config config.yaml --gpus "2,3"
# =============================================================================

# 设置默认参数
DEFAULT_CONFIG=""
DEFAULT_EXPERIMENT=""
DEFAULT_GPUS="2,3"

# 初始化参数
CONFIG_FILE="$DEFAULT_CONFIG"
EXPERIMENT="$DEFAULT_EXPERIMENT"
GPUS="$DEFAULT_GPUS"
USE_CONFIG=false========================================================================
# 训练退化恢复模型脚本 (支持YAML配置)
# 使用方法: 
#   1. 使用YAML配置文件: ./run_train_dc.sh --config config.yaml
#   2. 使用YAML配置文件和指定实验: ./run_train_dc.sh --config config.yaml --experiment small_model
#   3. 使用YAML配置文件并指定GPU: ./run_train_dc.sh --config config.yaml --gpus "2,3"
# =============================================================================

# 设置默认参数
DEFAULT_CONFIG=""
DEFAULT_EXPERIMENT=""
DEFAULT_GPUS="2,3"

# 初始化参数
CONFIG_FILE=$DEFAULT_CONFIG
EXPERIMENT=$DEFAULT_EXPERIMENT
GPUS=$DEFAULT_GPUS
USE_CONFIG=false

# 帮助信息
show_help() {
    cat << EOF
训练退化恢复模型脚本 (支持YAML配置)

使用方法: $0 [选项]

主要使用方式:
    1. 使用YAML配置文件:
       $0 --config config.yaml
    
    2. 使用YAML配置文件和指定实验:
       $0 --config config.yaml --experiment small_model
    
    3. 使用YAML配置文件并指定GPU:
       $0 --config config.yaml --gpus "2,3"

配置选项:
    --config FILE           YAML配置文件路径
    --experiment NAME       实验名称（在YAML配置中定义）
    --gpus GPUS            使用的GPU列表,用逗号分隔 (默认: $DEFAULT_GPUS)
    -h, --help             显示此帮助信息

实验示例:
    # 使用默认配置
    $0 --config config.yaml

    # 使用小模型实验配置
    $0 --config config.yaml --experiment small_model

    # 使用指定GPU
    $0 --config config.yaml --gpus "2,3"

EOF
}

# 解析命令行参数
while [ $# -gt 0 ]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            USE_CONFIG=true
            shift 2
            ;;
        --experiment)
            EXPERIMENT="$2"
            shift 2
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看帮助信息"
            exit 1
            ;;
    esac
done

# 验证参数
if [ "$USE_CONFIG" = "true" ]; then
    if [ -z "$CONFIG_FILE" ]; then
        echo "错误: 请指定配置文件路径"
        exit 1
    fi
    
    if [ ! -f "$CONFIG_FILE" ]; then
        echo "错误: 配置文件不存在: $CONFIG_FILE"
        exit 1
    fi
    
    echo "🚀 开始训练退化恢复模型 (使用YAML配置)"
    echo "================================="
    echo "📄 配置文件: $CONFIG_FILE"
    if [ -n "$EXPERIMENT" ]; then
        echo "🧪 实验配置: $EXPERIMENT"
    fi
    echo "🎮 GPU设备: $GPUS"
    echo "================================="
    
    # 构建Python命令参数
    PYTHON_ARGS="--config \"$CONFIG_FILE\""
    if [ -n "$EXPERIMENT" ]; then
        PYTHON_ARGS="$PYTHON_ARGS --experiment \"$EXPERIMENT\""
    fi
    
    # 设置CUDA可见设备
    export CUDA_VISIBLE_DEVICES="$GPUS"
    
    # 执行训练命令
    echo "🔥 执行命令: CUDA_VISIBLE_DEVICES=$GPUS python train_dc.py $PYTHON_ARGS"
    echo "================================="
    
    eval "python train_dc.py $PYTHON_ARGS"
    
    # 检查执行结果
    if [ $? -eq 0 ]; then
        echo "✅ 训练完成!"
    else
        echo "❌ 训练失败，请检查错误信息"
        exit 1
    fi
else
    echo "❌ 错误: 请使用YAML配置文件"
    echo "使用方法: $0 --config config.yaml"
    echo "使用 --help 查看更多帮助信息"
    exit 1
fi