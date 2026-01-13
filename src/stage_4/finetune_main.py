#!/usr/bin/env python3
"""
本地 LLM 微调主入口

提供命令行界面进行：
- 一键微调
- 模型测试
- 交互式对话

Usage:
    # 一键微调
    python -m src.stage_4.finetune_main train
    
    # 使用微调后的模型对话
    python -m src.stage_4.finetune_main chat
    
    # 测试模型效果
    python -m src.stage_4.finetune_main test
    
    # 查看帮助
    python -m src.stage_4.finetune_main --help
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional

from loguru import logger

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def setup_logging(verbose: bool = False):
    """配置日志"""
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level=level,
    )


def train_model(
    data_path: str,
    model: str,
    output_dir: str,
    epochs: int,
    lora_rank: int,
    batch_size: int,
    max_seq_length: int,
    device: str,
    learning_rate: float,
):
    """执行模型微调"""
    from src.stage_4.fine_tuning import LocalLLMFineTuner, LocalFineTuneConfig
    
    print("\n" + "=" * 60)
    print("🚀 本地 LLM 微调")
    print("=" * 60)
    
    # 检查数据文件
    if not os.path.exists(data_path):
        print(f"\n❌ 训练数据不存在: {data_path}")
        print("请先生成训练数据，或指定正确的数据路径")
        return False
    
    # 创建配置
    config = LocalFineTuneConfig(
        base_model=model,
        output_dir=output_dir,
        lora_rank=lora_rank,
        epochs=epochs,
        batch_size=batch_size,
        max_seq_length=max_seq_length,
        device=device,
        learning_rate=learning_rate,
    )
    
    # 显示配置
    print(f"\n📋 训练配置:")
    print(f"   - 基础模型: {config.base_model}")
    print(f"   - 训练数据: {data_path}")
    print(f"   - 输出目录: {config.output_dir}")
    print(f"   - LoRA rank: {config.lora_rank}")
    print(f"   - 训练轮数: {config.epochs}")
    print(f"   - 批次大小: {config.batch_size}")
    print(f"   - 序列长度: {config.max_seq_length}")
    print(f"   - 学习率: {config.learning_rate}")
    print(f"   - 训练设备: {config.device}")
    
    # 确认开始
    print("\n⚠️ 训练可能需要较长时间，请确保：")
    print("   1. 有足够的磁盘空间（约 5GB）")
    print("   2. 内存充足（建议 16GB+）")
    print("   3. 首次运行需要下载模型（约 1-3GB）")
    
    confirm = input("\n是否开始训练？(y/n): ").strip().lower()
    if confirm != 'y':
        print("已取消训练")
        return False
    
    # 开始微调
    try:
        finetuner = LocalLLMFineTuner(config)
        result = finetuner.run_full_pipeline(data_path)
        
        print("\n" + "=" * 60)
        print("✅ 微调完成!")
        print("=" * 60)
        print(f"\n📁 模型保存位置: {result['adapter_path']}")
        print(f"📊 训练损失: {result['metrics'].get('train_loss', 'N/A')}")
        print(f"\n🧪 测试回复: {result['test_response'][:200]}...")
        
        print("\n💡 下一步:")
        print(f"   运行 `python -m src.stage_4.finetune_main chat` 开始对话")
        
        return True
        
    except Exception as e:
        print(f"\n❌ 微调失败: {e}")
        logger.exception("微调过程出错")
        return False


def interactive_chat(
    model_path: Optional[str],
    base_model: str,
    device: str,
):
    """交互式对话"""
    from src.stage_4.fine_tuning import LocalLLMFineTuner, LocalFineTuneConfig
    
    print("\n" + "=" * 60)
    print("💬 微调模型对话")
    print("=" * 60)
    
    # 确定模型路径
    adapter_path = model_path or "./models/finetuned_llm/adapter"
    
    if not os.path.exists(adapter_path):
        print(f"\n❌ 模型适配器不存在: {adapter_path}")
        print("请先运行微调，或指定正确的模型路径")
        return
    
    print(f"\n📂 加载模型适配器: {adapter_path}")
    print(f"📦 基础模型: {base_model}")
    
    # 创建微调器并加载模型
    config = LocalFineTuneConfig(
        base_model=base_model,
        device=device,
    )
    
    try:
        finetuner = LocalLLMFineTuner(config)
        finetuner.load_adapter(adapter_path, base_model)
        
        print(f"\n✅ 模型加载完成!")
        print(f"   - 设备: {finetuner.device}")
        
    except Exception as e:
        print(f"\n❌ 模型加载失败: {e}")
        return
    
    # 交互式对话
    print("\n" + "-" * 60)
    print("开始对话（输入 'quit' 退出，'clear' 清屏）")
    print("-" * 60)
    
    while True:
        try:
            question = input("\n🧑 你: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("\n👋 再见！")
                break
            
            if question.lower() == 'clear':
                os.system('clear' if os.name != 'nt' else 'cls')
                continue
            
            # 生成回复
            print("\n🤖 助手: ", end="", flush=True)
            response = finetuner.chat(
                question,
                max_new_tokens=512,
                temperature=0.7,
            )
            print(response)
            
        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"\n❌ 生成回复时出错: {e}")


def test_model(
    model_path: Optional[str],
    base_model: str,
    device: str,
    test_questions: Optional[list] = None,
):
    """测试模型效果"""
    from src.stage_4.fine_tuning import LocalLLMFineTuner, LocalFineTuneConfig
    
    print("\n" + "=" * 60)
    print("🧪 模型测试")
    print("=" * 60)
    
    # 确定模型路径
    adapter_path = model_path or "./models/finetuned_llm/adapter"
    
    if not os.path.exists(adapter_path):
        print(f"\n❌ 模型适配器不存在: {adapter_path}")
        return
    
    # 默认测试问题
    if test_questions is None:
        test_questions = [
            "小米公司是什么时候成立的？创始人是谁？",
            "小米的主要业务包括哪些？",
            "小米在2024年的营业额是多少？",
            "小米公司的总部在哪里？",
            "小米什么时候宣布进入造车领域的？",
        ]
    
    print(f"\n📂 加载模型: {adapter_path}")
    
    # 加载模型
    config = LocalFineTuneConfig(
        base_model=base_model,
        device=device,
    )
    
    try:
        finetuner = LocalLLMFineTuner(config)
        finetuner.load_adapter(adapter_path, base_model)
        print(f"✅ 模型加载完成")
    except Exception as e:
        print(f"\n❌ 模型加载失败: {e}")
        return
    
    # 测试问答
    print("\n" + "-" * 60)
    print("📝 测试结果")
    print("-" * 60)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n【问题 {i}】{question}")
        
        try:
            response = finetuner.chat(
                question,
                max_new_tokens=256,
                temperature=0.7,
            )
            print(f"【回答】{response}")
        except Exception as e:
            print(f"【错误】{e}")
        
        print("-" * 40)
    
    print("\n✅ 测试完成!")


def compare_models(
    base_model: str,
    adapter_path: str,
    device: str,
):
    """对比微调前后效果"""
    from src.stage_4.fine_tuning import LocalLLMFineTuner, LocalFineTuneConfig
    
    print("\n" + "=" * 60)
    print("📊 微调前后效果对比")
    print("=" * 60)
    
    test_questions = [
        "小米公司是什么时候成立的？",
        "小米的创始人是谁？",
    ]
    
    config = LocalFineTuneConfig(
        base_model=base_model,
        device=device,
    )
    
    # 加载原始模型
    print("\n📦 加载原始模型...")
    finetuner_base = LocalLLMFineTuner(config)
    finetuner_base.load_model()
    
    # 加载微调模型
    print("\n📦 加载微调模型...")
    finetuner_tuned = LocalLLMFineTuner(config)
    finetuner_tuned.load_adapter(adapter_path, base_model)
    
    # 对比测试
    print("\n" + "-" * 60)
    
    for question in test_questions:
        print(f"\n【问题】{question}")
        print()
        
        # 原始模型回答
        try:
            response_base = finetuner_base.chat(question, max_new_tokens=200)
            print(f"🔹 原始模型: {response_base[:300]}...")
        except Exception as e:
            print(f"🔹 原始模型: 错误 - {e}")
        
        print()
        
        # 微调模型回答
        try:
            response_tuned = finetuner_tuned.chat(question, max_new_tokens=200)
            print(f"🔸 微调模型: {response_tuned[:300]}...")
        except Exception as e:
            print(f"🔸 微调模型: 错误 - {e}")
        
        print("-" * 40)


def generate_training_data(
    data_dir: str,
    output_dir: str,
    pairs_per_doc: int,
):
    """生成训练数据"""
    from src.stage_1.document_loader import DocumentLoader
    from src.stage_1.chunker import TextChunker
    from src.stage_4.fine_tuning import LLMFineTuner
    from src.stage_4.config import get_stage4_config
    
    print("\n" + "=" * 60)
    print("📝 生成训练数据")
    print("=" * 60)
    
    # 加载文档
    print(f"\n📂 加载文档: {data_dir}")
    loader = DocumentLoader()
    documents = loader.load_directory(data_dir)
    
    if not documents:
        print(f"❌ 没有找到文档")
        return
    
    print(f"   找到 {len(documents)} 个文档")
    
    # 分块
    print("\n✂️ 文档分块...")
    chunker = TextChunker()
    chunks = chunker.split_documents(documents)
    print(f"   生成 {len(chunks)} 个文档块")
    
    # 生成训练数据
    print(f"\n🔄 生成 QA 对 (每个文档 {pairs_per_doc} 对)...")
    config = get_stage4_config()
    finetuner = LLMFineTuner(config=config, output_dir=output_dir)
    
    qa_pairs = finetuner.generate_qa_pairs(chunks, pairs_per_doc=pairs_per_doc)
    
    print(f"   生成了 {len(qa_pairs)} 个 QA 对")
    
    # 保存
    print("\n💾 保存数据...")
    finetuner.export_json()  # Alpaca 格式
    finetuner.export_jsonl()  # OpenAI 格式
    
    print(f"\n✅ 数据已保存到: {output_dir}")
    print(f"   - train_alpaca.json (Alpaca 格式)")
    print(f"   - train_openai.jsonl (OpenAI 格式)")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="UltimateRAG 本地 LLM 微调工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认配置进行微调
  python -m src.stage_4.finetune_main train
  
  # 指定模型和参数
  python -m src.stage_4.finetune_main train --model Qwen/Qwen2.5-0.5B-Instruct --epochs 3
  
  # 使用微调后的模型对话
  python -m src.stage_4.finetune_main chat
  
  # 测试模型效果
  python -m src.stage_4.finetune_main test
  
  # 生成训练数据
  python -m src.stage_4.finetune_main generate
  
  # 对比微调前后效果
  python -m src.stage_4.finetune_main compare
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # ============== train 命令 ==============
    train_parser = subparsers.add_parser("train", help="微调模型")
    train_parser.add_argument(
        "--data", "-d",
        default="./data/finetune/train_alpaca.json",
        help="训练数据路径 (默认: ./data/finetune/train_alpaca.json)"
    )
    train_parser.add_argument(
        "--model", "-m",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="基础模型 (默认: Qwen/Qwen2.5-0.5B-Instruct)"
    )
    train_parser.add_argument(
        "--output", "-o",
        default="./models/finetuned_llm",
        help="输出目录 (默认: ./models/finetuned_llm)"
    )
    train_parser.add_argument(
        "--epochs", "-e",
        type=int,
        default=3,
        help="训练轮数 (默认: 3)"
    )
    train_parser.add_argument(
        "--lora-rank", "-r",
        type=int,
        default=8,
        help="LoRA rank (默认: 8)"
    )
    train_parser.add_argument(
        "--batch-size", "-b",
        type=int,
        default=1,
        help="批次大小 (默认: 1)"
    )
    train_parser.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
        help="最大序列长度 (默认: 512)"
    )
    train_parser.add_argument(
        "--learning-rate", "-lr",
        type=float,
        default=2e-4,
        help="学习率 (默认: 2e-4)"
    )
    train_parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="训练设备 (默认: auto)"
    )
    
    # ============== chat 命令 ==============
    chat_parser = subparsers.add_parser("chat", help="与微调后的模型对话")
    chat_parser.add_argument(
        "--model-path", "-p",
        default=None,
        help="模型适配器路径 (默认: ./models/finetuned_llm/adapter)"
    )
    chat_parser.add_argument(
        "--base-model", "-m",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="基础模型 (默认: Qwen/Qwen2.5-0.5B-Instruct)"
    )
    chat_parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="推理设备 (默认: auto)"
    )
    
    # ============== test 命令 ==============
    test_parser = subparsers.add_parser("test", help="测试模型效果")
    test_parser.add_argument(
        "--model-path", "-p",
        default=None,
        help="模型适配器路径"
    )
    test_parser.add_argument(
        "--base-model", "-m",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="基础模型"
    )
    test_parser.add_argument(
        "--device",
        default="auto",
        help="推理设备"
    )
    
    # ============== compare 命令 ==============
    compare_parser = subparsers.add_parser("compare", help="对比微调前后效果")
    compare_parser.add_argument(
        "--model-path", "-p",
        default="./models/finetuned_llm/adapter",
        help="微调模型适配器路径"
    )
    compare_parser.add_argument(
        "--base-model", "-m",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="基础模型"
    )
    compare_parser.add_argument(
        "--device",
        default="auto",
        help="推理设备"
    )
    
    # ============== generate 命令 ==============
    gen_parser = subparsers.add_parser("generate", help="生成训练数据")
    gen_parser.add_argument(
        "--data-dir", "-d",
        default="./data/documents",
        help="文档目录 (默认: ./data/documents)"
    )
    gen_parser.add_argument(
        "--output", "-o",
        default="./data/finetune",
        help="输出目录 (默认: ./data/finetune)"
    )
    gen_parser.add_argument(
        "--pairs-per-doc",
        type=int,
        default=5,
        help="每个文档生成的 QA 对数量 (默认: 5)"
    )
    
    # 全局参数
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="显示详细日志"
    )
    
    args = parser.parse_args()
    
    # 配置日志
    setup_logging(getattr(args, 'verbose', False))
    
    # 执行命令
    if args.command == "train":
        train_model(
            data_path=args.data,
            model=args.model,
            output_dir=args.output,
            epochs=args.epochs,
            lora_rank=args.lora_rank,
            batch_size=args.batch_size,
            max_seq_length=args.max_seq_length,
            device=args.device,
            learning_rate=args.learning_rate,
        )
    
    elif args.command == "chat":
        interactive_chat(
            model_path=args.model_path,
            base_model=args.base_model,
            device=args.device,
        )
    
    elif args.command == "test":
        test_model(
            model_path=args.model_path,
            base_model=args.base_model,
            device=args.device,
        )
    
    elif args.command == "compare":
        compare_models(
            base_model=args.base_model,
            adapter_path=args.model_path,
            device=args.device,
        )
    
    elif args.command == "generate":
        generate_training_data(
            data_dir=args.data_dir,
            output_dir=args.output,
            pairs_per_doc=args.pairs_per_doc,
        )
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

