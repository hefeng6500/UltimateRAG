#!/usr/bin/env python3
"""
本地 LLM 微调测试脚本

用于验证微调环境是否正确配置。

Usage:
    python scripts/test_local_finetune.py
"""

import sys
import os

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_imports():
    """测试依赖导入"""
    print("=" * 50)
    print("📦 测试依赖导入...")
    print("=" * 50)
    
    errors = []
    
    # 测试 torch
    try:
        import torch
        print(f"✅ torch: {torch.__version__}")
        print(f"   - CUDA 可用: {torch.cuda.is_available()}")
        print(f"   - MPS 可用: {torch.backends.mps.is_available()}")
    except ImportError as e:
        errors.append(f"❌ torch: {e}")
    
    # 测试 transformers
    try:
        import transformers
        print(f"✅ transformers: {transformers.__version__}")
    except ImportError as e:
        errors.append(f"❌ transformers: {e}")
    
    # 测试 peft
    try:
        import peft
        print(f"✅ peft: {peft.__version__}")
    except ImportError as e:
        errors.append(f"❌ peft: {e}")
    
    # 测试 datasets
    try:
        import datasets
        print(f"✅ datasets: {datasets.__version__}")
    except ImportError as e:
        errors.append(f"❌ datasets: {e}")
    
    # 测试 accelerate
    try:
        import accelerate
        print(f"✅ accelerate: {accelerate.__version__}")
    except ImportError as e:
        errors.append(f"❌ accelerate: {e}")
    
    if errors:
        print("\n⚠️ 发现以下错误:")
        for err in errors:
            print(f"   {err}")
        print("\n请运行: pip install torch transformers peft datasets accelerate trl")
        return False
    
    return True


def test_module_import():
    """测试模块导入"""
    print("\n" + "=" * 50)
    print("🔧 测试模块导入...")
    print("=" * 50)
    
    try:
        from src.stage_4.fine_tuning import (
            LocalLLMFineTuner,
            LocalFineTuneConfig,
            quick_finetune,
        )
        print("✅ LocalLLMFineTuner 导入成功")
        print("✅ LocalFineTuneConfig 导入成功")
        print("✅ quick_finetune 导入成功")
        return True
    except ImportError as e:
        print(f"❌ 模块导入失败: {e}")
        return False


def test_config():
    """测试配置创建"""
    print("\n" + "=" * 50)
    print("⚙️ 测试配置...")
    print("=" * 50)
    
    try:
        from src.stage_4.fine_tuning import LocalFineTuneConfig
        
        config = LocalFineTuneConfig(
            base_model="Qwen/Qwen2.5-0.5B-Instruct",
            output_dir="./models/test_finetune",
            lora_rank=4,
            epochs=1,
            device="cpu",
        )
        
        print(f"✅ 配置创建成功:")
        print(f"   - 基础模型: {config.base_model}")
        print(f"   - 输出目录: {config.output_dir}")
        print(f"   - LoRA rank: {config.lora_rank}")
        print(f"   - 训练轮数: {config.epochs}")
        print(f"   - 训练设备: {config.device}")
        return True
    except Exception as e:
        print(f"❌ 配置创建失败: {e}")
        return False


def test_finetuner_init():
    """测试微调器初始化"""
    print("\n" + "=" * 50)
    print("🚀 测试微调器初始化...")
    print("=" * 50)
    
    try:
        from src.stage_4.fine_tuning import LocalLLMFineTuner, LocalFineTuneConfig
        
        config = LocalFineTuneConfig(
            base_model="Qwen/Qwen2.5-0.5B-Instruct",
            device="cpu",
        )
        
        finetuner = LocalLLMFineTuner(config)
        print(f"✅ 微调器初始化成功")
        print(f"   - 设备: {finetuner.device}")
        return True
    except Exception as e:
        print(f"❌ 微调器初始化失败: {e}")
        return False


def test_data_exists():
    """测试训练数据是否存在"""
    print("\n" + "=" * 50)
    print("📂 测试训练数据...")
    print("=" * 50)
    
    data_path = "./data/finetune/train_alpaca.json"
    
    if os.path.exists(data_path):
        import json
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"✅ 训练数据存在: {data_path}")
        print(f"   - 样本数: {len(data)}")
        return True
    else:
        print(f"⚠️ 训练数据不存在: {data_path}")
        print("   请先运行数据生成命令生成训练数据")
        return False


def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("   🧪 UltimateRAG 本地 LLM 微调环境测试")
    print("=" * 60)
    
    results = []
    
    # 运行测试
    results.append(("依赖导入", test_imports()))
    results.append(("模块导入", test_module_import()))
    results.append(("配置测试", test_config()))
    results.append(("微调器初始化", test_finetuner_init()))
    results.append(("训练数据", test_data_exists()))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("   📊 测试结果汇总")
    print("=" * 60)
    
    passed = 0
    failed = 0
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"   {name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print(f"\n   总计: {passed} 通过, {failed} 失败")
    
    if failed == 0:
        print("\n🎉 所有测试通过！可以开始微调了。")
        print("\n快速开始:")
        print(">>> from src.stage_4.fine_tuning import quick_finetune")
        print(">>> quick_finetune(model='Qwen/Qwen2.5-0.5B-Instruct', epochs=1)")
    else:
        print("\n⚠️ 部分测试失败，请根据上述提示修复问题。")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

