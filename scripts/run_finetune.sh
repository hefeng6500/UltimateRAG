#!/bin/bash
#
# UltimateRAG 本地 LLM 微调快速启动脚本
#
# Usage:
#   ./scripts/run_finetune.sh train    # 微调模型
#   ./scripts/run_finetune.sh chat     # 交互式对话
#   ./scripts/run_finetune.sh test     # 测试模型
#   ./scripts/run_finetune.sh all      # 完整流程
#

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 默认配置
MODEL="Qwen/Qwen2.5-0.5B-Instruct"
DATA_PATH="./data/finetune/train_alpaca.json"
OUTPUT_DIR="./models/finetuned_llm"
EPOCHS=3
LORA_RANK=8

# 切换到项目根目录
cd "$(dirname "$0")/.."

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 显示帮助信息
show_help() {
    echo ""
    echo "=========================================="
    echo "  UltimateRAG 本地 LLM 微调工具"
    echo "=========================================="
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  train       微调模型"
    echo "  chat        使用微调后的模型进行交互式对话"
    echo "  test        测试模型效果"
    echo "  compare     对比微调前后效果"
    echo "  generate    生成训练数据"
    echo "  check       检查环境配置"
    echo "  all         完整流程 (check -> train -> test -> chat)"
    echo "  help        显示此帮助信息"
    echo ""
    echo "Examples:"
    echo "  $0 train                    # 使用默认配置微调"
    echo "  $0 chat                     # 开始对话"
    echo "  $0 all                      # 运行完整流程"
    echo ""
    echo "Environment Variables:"
    echo "  MODEL       基础模型 (默认: $MODEL)"
    echo "  EPOCHS      训练轮数 (默认: $EPOCHS)"
    echo "  LORA_RANK   LoRA rank (默认: $LORA_RANK)"
    echo ""
}

# 检查环境
check_env() {
    print_info "检查环境配置..."
    python scripts/test_local_finetune.py
}

# 微调模型
train_model() {
    print_info "开始微调模型..."
    print_info "模型: $MODEL"
    print_info "训练数据: $DATA_PATH"
    print_info "训练轮数: $EPOCHS"
    print_info "LoRA rank: $LORA_RANK"
    echo ""
    
    python -m src.stage_4.finetune_main train \
        --model "$MODEL" \
        --data "$DATA_PATH" \
        --output "$OUTPUT_DIR" \
        --epochs "$EPOCHS" \
        --lora-rank "$LORA_RANK"
}

# 交互式对话
chat_model() {
    print_info "启动交互式对话..."
    python -m src.stage_4.finetune_main chat \
        --base-model "$MODEL"
}

# 测试模型
test_model() {
    print_info "测试模型效果..."
    python -m src.stage_4.finetune_main test \
        --base-model "$MODEL"
}

# 对比模型
compare_model() {
    print_info "对比微调前后效果..."
    python -m src.stage_4.finetune_main compare \
        --base-model "$MODEL"
}

# 生成数据
generate_data() {
    print_info "生成训练数据..."
    python -m src.stage_4.finetune_main generate
}

# 完整流程
run_all() {
    echo ""
    echo "=========================================="
    echo "  运行完整微调流程"
    echo "=========================================="
    echo ""
    
    # Step 1: 检查环境
    print_info "Step 1/4: 检查环境"
    check_env
    echo ""
    
    # Step 2: 微调
    print_info "Step 2/4: 微调模型"
    train_model
    echo ""
    
    # Step 3: 测试
    print_info "Step 3/4: 测试模型"
    test_model
    echo ""
    
    # Step 4: 对话
    print_info "Step 4/4: 交互式对话"
    chat_model
}

# 主函数
main() {
    # 如果设置了环境变量，使用环境变量
    MODEL=${MODEL:-"Qwen/Qwen2.5-0.5B-Instruct"}
    EPOCHS=${EPOCHS:-3}
    LORA_RANK=${LORA_RANK:-8}
    
    case "${1:-help}" in
        train)
            train_model
            ;;
        chat)
            chat_model
            ;;
        test)
            test_model
            ;;
        compare)
            compare_model
            ;;
        generate)
            generate_data
            ;;
        check)
            check_env
            ;;
        all)
            run_all
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            print_error "未知命令: $1"
            show_help
            exit 1
            ;;
    esac
}

main "$@"

