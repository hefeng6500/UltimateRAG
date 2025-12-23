# 本地 LLM 微调指南

## 📖 概述

本模块提供了在本地设备上进行 LLM 微调的完整解决方案。使用 **LoRA (Low-Rank Adaptation)** 技术，可以在有限的硬件资源下高效微调大语言模型。

### 核心特性

| 特性 | 说明 |
|------|------|
| **低资源需求** | LoRA 仅训练约 0.1%-1% 的参数，大幅降低内存需求 |
| **多设备支持** | 支持 CPU / CUDA / MPS (Apple Silicon) |
| **主流模型** | 支持 Qwen、LLaMA、Mistral 等开源模型 |
| **数据复用** | 直接使用已生成的 Alpaca 格式训练数据 |
| **一键微调** | 提供命令行工具和 Python API |
| **交互对话** | 微调完成后可直接进行交互式问答 |

---

## 🖥️ 硬件要求

### 最低配置

| 配置项 | CPU 训练 | GPU 训练 |
|--------|----------|----------|
| 内存/显存 | 16GB RAM | 8GB VRAM |
| 推荐模型 | 0.5B-1.5B | 1.5B-7B |
| 训练速度 | 较慢 | 快 |

### 推荐模型选择

| 硬件配置 | 推荐模型 | 预估时间/epoch |
|----------|----------|----------------|
| 16GB RAM (CPU) | Qwen2.5-0.5B-Instruct | ~30分钟 |
| 16GB RAM (CPU) | Qwen2.5-1.5B-Instruct | ~60分钟 |
| 8GB VRAM (GPU) | Qwen2.5-1.5B-Instruct | ~5分钟 |
| 16GB VRAM (GPU) | Qwen2.5-7B-Instruct | ~10分钟 |

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install torch transformers peft datasets accelerate trl
```

或使用项目的 requirements.txt：

```bash
pip install -r requirements.txt
```

### 2. 命令行一键微调 ⭐

**最简单的方式 - 一行命令完成微调：**

```bash
# 使用默认配置进行微调
python -m src.stage_4.finetune_main train

# 指定模型和参数
python -m src.stage_4.finetune_main train --model Qwen/Qwen2.5-0.5B-Instruct --epochs 3

# 查看所有可用参数
python -m src.stage_4.finetune_main train --help
```

### 3. 使用微调后的模型对话 ⭐

```bash
# 启动交互式对话
python -m src.stage_4.finetune_main chat

# 指定模型路径
python -m src.stage_4.finetune_main chat --model-path ./models/finetuned_llm/adapter
```

### 4. 测试模型效果

```bash
# 运行预设测试问题
python -m src.stage_4.finetune_main test

# 对比微调前后效果
python -m src.stage_4.finetune_main compare
```

### 5. Python API 方式

```python
from src.stage_4.fine_tuning import quick_finetune

# 使用默认配置进行微调
result = quick_finetune(
    data_path="./data/finetune/train_alpaca.json",
    model="Qwen/Qwen2.5-0.5B-Instruct",
    epochs=3
)

print(f"训练损失: {result['metrics']['train_loss']}")
print(f"测试回复: {result['test_response']}")
```

### 6. 使用微调后的模型 (Python)

```python
from src.stage_4.fine_tuning import LocalLLMFineTuner

# 加载微调后的模型
finetuner = LocalLLMFineTuner()
finetuner.load_adapter(
    adapter_path="./models/finetuned_llm/adapter",
    base_model_path="Qwen/Qwen2.5-0.5B-Instruct"
)

# 进行对话
response = finetuner.chat("小米公司是什么时候成立的？")
print(response)
```

---

## 📚 详细使用指南

### 完整微调流程

```python
from src.stage_4.fine_tuning import LocalLLMFineTuner, LocalFineTuneConfig

# 1. 创建配置
config = LocalFineTuneConfig(
    # 模型配置
    base_model="Qwen/Qwen2.5-1.5B-Instruct",
    output_dir="./models/my_finetuned_model",
    
    # LoRA 配置
    lora_rank=8,          # 秩越小，参数越少，推荐 4-16
    lora_alpha=16,        # 通常设为 2 * lora_rank
    lora_dropout=0.05,
    
    # 训练配置
    epochs=3,
    batch_size=1,         # CPU 建议用 1
    gradient_accumulation_steps=8,  # 累积 8 步 = 有效批次 8
    learning_rate=2e-4,
    max_seq_length=512,
    
    # 设备配置
    device="cpu",         # 或 "auto" 自动检测
)

# 2. 初始化微调器
finetuner = LocalLLMFineTuner(config)

# 3. 加载模型
finetuner.load_model()

# 4. 配置 LoRA
finetuner.setup_lora()

# 5. 加载训练数据
finetuner.load_data(
    data_path="./data/finetune/train_alpaca.json",
    eval_ratio=0.1,       # 10% 作为验证集
    data_format="alpaca"  # 数据格式
)

# 6. 开始训练
metrics = finetuner.train()

# 7. 保存 LoRA 适配器
adapter_path = finetuner.save()

# 8. (可选) 合并到基础模型
merged_path = finetuner.merge_and_save()

# 9. 测试效果
response = finetuner.chat("请介绍一下小米公司的发展历程")
print(response)
```

---

## ⚙️ 配置详解

### LocalFineTuneConfig 参数

```python
@dataclass
class LocalFineTuneConfig:
    # === 模型配置 ===
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    # HuggingFace 模型 ID 或本地路径
    # 推荐模型:
    #   - Qwen/Qwen2.5-0.5B-Instruct (最小)
    #   - Qwen/Qwen2.5-1.5B-Instruct (推荐)
    #   - Qwen/Qwen2.5-3B-Instruct
    #   - meta-llama/Llama-3.2-1B-Instruct
    #   - mistralai/Mistral-7B-Instruct-v0.3
    
    output_dir: str = "./models/finetuned_llm"
    # 微调输出目录
    
    # === LoRA 配置 ===
    lora_rank: int = 8
    # LoRA 秩，越小参数越少
    # 推荐: 4(超低资源) / 8(推荐) / 16(效果更好)
    
    lora_alpha: int = 16
    # LoRA alpha，通常设为 2 * rank
    
    lora_dropout: float = 0.05
    # Dropout 比例，防止过拟合
    
    target_modules: Optional[List[str]] = None
    # 要微调的模块，None 则自动检测
    # Qwen/LLaMA: ["q_proj", "k_proj", "v_proj", "o_proj"]
    
    # === 训练配置 ===
    epochs: int = 3
    # 训练轮数，小数据集建议 3-5 轮
    
    batch_size: int = 1
    # 批次大小，CPU 建议 1，GPU 可增大
    
    gradient_accumulation_steps: int = 8
    # 梯度累积步数
    # 有效批次 = batch_size * gradient_accumulation_steps
    
    learning_rate: float = 2e-4
    # 学习率，LoRA 通常用较大学习率
    
    max_seq_length: int = 512
    # 最大序列长度，影响内存使用
    
    warmup_ratio: float = 0.1
    # 学习率预热比例
    
    # === 设备配置 ===
    device: str = "auto"
    # 训练设备: auto / cpu / cuda / mps
    # auto 会自动检测最佳设备
```

---

## 📁 数据格式

### 支持的格式

#### 1. Alpaca 格式 (推荐)

```json
[
  {
    "instruction": "问题或指令",
    "input": "可选的上下文",
    "output": "期望的回答"
  }
]
```

#### 2. OpenAI 格式

```jsonl
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

#### 3. 原始文本格式

```json
[
  {"text": "完整的对话文本..."}
]
```

### 使用已生成的数据

项目中已经生成了训练数据，可以直接使用：

```python
# 使用 Alpaca 格式数据
finetuner.load_data(
    data_path="./data/finetune/train_alpaca.json",
    data_format="alpaca"
)

# 使用 OpenAI 格式数据
finetuner.load_data(
    data_path="./data/finetune/train_openai.jsonl",
    data_format="openai"
)
```

---

## 💡 最佳实践

### 1. 内存优化

```python
config = LocalFineTuneConfig(
    # 使用较小的模型
    base_model="Qwen/Qwen2.5-0.5B-Instruct",
    
    # 减小 LoRA 秩
    lora_rank=4,
    
    # 减小序列长度
    max_seq_length=256,
    
    # 减小批次，增加梯度累积
    batch_size=1,
    gradient_accumulation_steps=16,
    
    # 启用梯度检查点
    gradient_checkpointing=True,
)
```

### 2. 提升效果

```python
config = LocalFineTuneConfig(
    # 增大 LoRA 秩
    lora_rank=16,
    lora_alpha=32,
    
    # 增加训练轮数
    epochs=5,
    
    # 使用更大的模型（需要更多资源）
    base_model="Qwen/Qwen2.5-3B-Instruct",
)
```

### 3. 数据质量

- **数据量**: 建议至少 100 条高质量数据
- **数据多样性**: 覆盖不同类型的问题
- **答案质量**: 确保答案准确、详细
- **去重**: 避免重复数据导致过拟合

---

## 🔧 常见问题

### Q1: 训练时内存不足 (OOM)

```python
# 解决方案 1: 减小批次
config.batch_size = 1
config.gradient_accumulation_steps = 16

# 解决方案 2: 减小序列长度
config.max_seq_length = 256

# 解决方案 3: 使用更小的模型
config.base_model = "Qwen/Qwen2.5-0.5B-Instruct"

# 解决方案 4: 减小 LoRA 秩
config.lora_rank = 4
```

### Q2: 训练速度太慢

```python
# CPU 训练本身较慢，可以:
# 1. 减少训练轮数
config.epochs = 2

# 2. 减少数据量（使用子集）
# 3. 使用更小的模型
# 4. 如果有 GPU，使用 GPU
config.device = "cuda"
```

### Q3: 模型效果不好

```python
# 1. 增加训练数据
# 2. 增加训练轮数
config.epochs = 5

# 3. 调整学习率
config.learning_rate = 1e-4  # 尝试更小的学习率

# 4. 增大 LoRA 秩
config.lora_rank = 16
```

### Q4: 如何在其他项目使用微调后的模型

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 方式 1: 使用 LoRA 适配器
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
model = PeftModel.from_pretrained(base_model, "./models/finetuned_llm/adapter")

# 方式 2: 使用合并后的模型
model = AutoModelForCausalLM.from_pretrained("./models/finetuned_llm/merged")
```

---

## 📊 训练监控

训练过程会输出以下信息：

```
🚀 开始训练...
训练配置:
   - 轮数: 3
   - 批次大小: 1
   - 梯度累积: 8
   - 有效批次: 8
   - 学习率: 0.0002

{'loss': 2.3456, 'learning_rate': 0.0001, 'epoch': 0.5}
{'loss': 1.8765, 'learning_rate': 0.0002, 'epoch': 1.0}
{'loss': 1.2345, 'learning_rate': 0.00015, 'epoch': 1.5}
...

✅ 训练完成!
   - 总步数: 150
   - 训练损失: 0.8765
   - 训练时间: 1800.5s
```

---

## 🔗 相关资源

- [LoRA 论文](https://arxiv.org/abs/2106.09685)
- [PEFT 官方文档](https://huggingface.co/docs/peft)
- [Qwen2.5 模型](https://huggingface.co/Qwen)
- [Transformers 微调指南](https://huggingface.co/docs/transformers/training)

---

## 🖥️ 命令行工具详解

### 所有命令一览

| 命令 | 说明 | 示例 |
|------|------|------|
| `train` | 微调模型 | `python -m src.stage_4.finetune_main train` |
| `chat` | 交互式对话 | `python -m src.stage_4.finetune_main chat` |
| `test` | 测试模型效果 | `python -m src.stage_4.finetune_main test` |
| `compare` | 对比微调前后 | `python -m src.stage_4.finetune_main compare` |
| `generate` | 生成训练数据 | `python -m src.stage_4.finetune_main generate` |

### train 命令参数

```bash
python -m src.stage_4.finetune_main train [OPTIONS]

Options:
  -d, --data PATH          训练数据路径 (默认: ./data/finetune/train_alpaca.json)
  -m, --model MODEL        基础模型 (默认: Qwen/Qwen2.5-0.5B-Instruct)
  -o, --output PATH        输出目录 (默认: ./models/finetuned_llm)
  -e, --epochs INT         训练轮数 (默认: 3)
  -r, --lora-rank INT      LoRA rank (默认: 8)
  -b, --batch-size INT     批次大小 (默认: 1)
  --max-seq-length INT     最大序列长度 (默认: 512)
  -lr, --learning-rate     学习率 (默认: 2e-4)
  --device DEVICE          训练设备: auto/cpu/cuda/mps (默认: auto)
```

### chat 命令参数

```bash
python -m src.stage_4.finetune_main chat [OPTIONS]

Options:
  -p, --model-path PATH    模型适配器路径 (默认: ./models/finetuned_llm/adapter)
  -m, --base-model MODEL   基础模型 (默认: Qwen/Qwen2.5-0.5B-Instruct)
  --device DEVICE          推理设备 (默认: auto)
```

### 完整工作流程示例

```bash
# Step 1: 检查环境
python scripts/test_local_finetune.py

# Step 2: (可选) 重新生成训练数据
python -m src.stage_4.finetune_main generate --data-dir ./data/documents

# Step 3: 开始微调
python -m src.stage_4.finetune_main train --model Qwen/Qwen2.5-0.5B-Instruct --epochs 3

# Step 4: 测试模型效果
python -m src.stage_4.finetune_main test

# Step 5: 交互式对话
python -m src.stage_4.finetune_main chat

# Step 6: (可选) 对比微调前后效果
python -m src.stage_4.finetune_main compare
```

---

## 📝 完整示例

### 命令行方式 (推荐)

```bash
# 完整的微调流程
python -m src.stage_4.finetune_main train \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --data ./data/finetune/train_alpaca.json \
    --output ./models/my_finetuned_model \
    --epochs 3 \
    --lora-rank 8

# 使用微调后的模型对话
python -m src.stage_4.finetune_main chat \
    --model-path ./models/my_finetuned_model/adapter \
    --base-model Qwen/Qwen2.5-0.5B-Instruct
```

### Python API 方式

```python
"""
完整的本地微调示例
"""
from src.stage_4.fine_tuning import (
    LocalLLMFineTuner,
    LocalFineTuneConfig,
    quick_finetune,
)

# ============== 方式 1: 快速微调 ==============
result = quick_finetune(
    data_path="./data/finetune/train_alpaca.json",
    model="Qwen/Qwen2.5-0.5B-Instruct",
    output_dir="./models/my_model",
    epochs=3,
)

# ============== 方式 2: 自定义配置 ==============
config = LocalFineTuneConfig(
    base_model="Qwen/Qwen2.5-1.5B-Instruct",
    output_dir="./models/custom_model",
    lora_rank=8,
    epochs=3,
    device="cpu",
)

finetuner = LocalLLMFineTuner(config)
finetuner.run_full_pipeline("./data/finetune/train_alpaca.json")

# ============== 方式 3: 分步执行 ==============
finetuner = LocalLLMFineTuner()
finetuner.load_model()
finetuner.setup_lora()
finetuner.load_data("./data/finetune/train_alpaca.json")
finetuner.train()
finetuner.save()

# 测试
response = finetuner.chat("你好，请介绍一下自己")
print(response)
```

---

## 🎬 快速演示

### 30 秒快速体验

```bash
# 1. 安装依赖 (首次)
pip install -r requirements.txt

# 2. 一键微调
python -m src.stage_4.finetune_main train --epochs 1

# 3. 开始对话
python -m src.stage_4.finetune_main chat
```

### 交互式对话示例

```
============================================================
💬 微调模型对话
============================================================

📂 加载模型适配器: ./models/finetuned_llm/adapter
📦 基础模型: Qwen/Qwen2.5-0.5B-Instruct

✅ 模型加载完成!
   - 设备: mps

------------------------------------------------------------
开始对话（输入 'quit' 退出，'clear' 清屏）
------------------------------------------------------------

🧑 你: 小米公司是什么时候成立的？

🤖 助手: 根据文档内容，小米科技有限责任公司成立于2010年3月3日，
总部位于北京市海淀区安宁庄路小米科技园，公司的创始人是雷军。

🧑 你: 小米的主要业务有哪些？

🤖 助手: 小米主要从事智能手机、智能汽车、芯片、物联网（IoT）以及
生活消费产品的研发和销售。此外，公司还提供互联网服务，并从事投资业务。

🧑 你: quit

👋 再见！
```

