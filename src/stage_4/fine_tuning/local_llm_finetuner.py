"""
本地 LLM 微调器

使用 Transformers + PEFT (LoRA) 在本地进行 LLM 微调。
支持 CPU 训练，适配 Apple Silicon 和普通 x86 设备。

主要特性：
- 支持 Qwen2.5、LLaMA、Mistral 等主流开源模型
- 使用 LoRA 低资源微调，显著降低内存需求
- 支持 CPU 训练（也支持 MPS/CUDA 如果可用）
- 复用现有的 Alpaca 格式训练数据
"""

import os
import json
from typing import Optional, List, Dict, Any, Union
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

# 延迟导入以避免未安装时报错
try:
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForSeq2Seq,
        BitsAndBytesConfig,
    )
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
        PeftModel,
        TaskType,
    )
    from datasets import Dataset
    HAS_TORCH = True
except ImportError as e:
    HAS_TORCH = False
    IMPORT_ERROR = str(e)


@dataclass
class LocalFineTuneConfig:
    """
    本地微调配置
    
    Attributes:
        base_model: HuggingFace 模型 ID 或本地路径
        output_dir: 微调模型输出目录
        lora_rank: LoRA 秩，越小越节省内存（推荐 4-16）
        lora_alpha: LoRA alpha 参数
        lora_dropout: LoRA dropout 比例
        target_modules: 要微调的模块（None 则自动检测）
        epochs: 训练轮数
        batch_size: 批次大小（CPU 建议 1-2）
        gradient_accumulation_steps: 梯度累积步数
        learning_rate: 学习率
        max_seq_length: 最大序列长度
        warmup_ratio: 学习率预热比例
        save_steps: 保存检查点的步数间隔
        logging_steps: 日志记录步数间隔
        device: 训练设备 (auto/cpu/cuda/mps)
    """
    # 模型配置
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    output_dir: str = "./models/finetuned_llm"
    
    # LoRA 配置
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    target_modules: Optional[List[str]] = None  # None = 自动检测
    
    # 训练配置
    epochs: int = 3
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    max_seq_length: int = 512
    warmup_ratio: float = 0.1
    save_steps: int = 100
    logging_steps: int = 10
    
    # 设备配置 (auto 会自动检测: MPS > CUDA > CPU)
    device: str = "auto"  # auto / cpu / cuda / mps
    
    # 优化配置
    fp16: bool = False  # CPU 不支持
    bf16: bool = False  # 需要特定硬件支持
    gradient_checkpointing: bool = True  # 节省内存


@dataclass
class ChatMessage:
    """对话消息"""
    role: str  # system / user / assistant
    content: str


class LocalLLMFineTuner:
    """
    本地 LLM 微调器
    
    使用 Transformers + PEFT (LoRA) 进行高效微调。
    
    Example:
        >>> finetuner = LocalLLMFineTuner()
        >>> finetuner.load_data("./data/finetune/train_alpaca.json")
        >>> finetuner.train()
        >>> finetuner.save()
        >>> response = finetuner.chat("小米公司是什么时候成立的？")
    """
    
    def __init__(self, config: Optional[LocalFineTuneConfig] = None):
        """
        初始化微调器
        
        Args:
            config: 微调配置，None 则使用默认配置
        """
        if not HAS_TORCH:
            raise ImportError(
                f"缺少必要依赖，请安装：\n"
                f"pip install torch transformers peft datasets accelerate trl\n"
                f"原始错误: {IMPORT_ERROR}"
            )
        
        self.config = config or LocalFineTuneConfig()
        self._setup_device()
        
        # 模型和分词器
        self.model = None
        self.tokenizer = None
        self.peft_model = None
        
        # 训练数据
        self.train_dataset = None
        self.eval_dataset = None
        
        logger.info(f"🔧 LocalLLMFineTuner 初始化完成")
        logger.info(f"   - 基础模型: {self.config.base_model}")
        logger.info(f"   - 输出目录: {self.config.output_dir}")
        logger.info(f"   - 训练设备: {self.device}")
    
    def _setup_device(self):
        """设置训练设备"""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = self.config.device
        
        logger.info(f"📱 使用设备: {self.device}")
        
        # CPU 特定配置
        if self.device == "cpu":
            logger.warning(
                "⚠️ 使用 CPU 训练，速度会较慢。"
                "建议使用较小的模型（如 Qwen2.5-0.5B 或 1.5B）"
            )
    
    def load_model(self, model_name_or_path: Optional[str] = None):
        """
        加载基础模型和分词器
        
        Args:
            model_name_or_path: 模型名称或路径，None 则使用配置中的模型
        """
        model_path = model_name_or_path or self.config.base_model
        logger.info(f"📥 加载模型: {model_path}")
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="right",
        )
        
        # 确保有 pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 加载模型
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float32 if self.device == "cpu" else torch.float16,
        }
        
        # CPU 加载到 CPU，其他设备使用 device_map
        if self.device == "cpu":
            model_kwargs["device_map"] = {"": "cpu"}
        else:
            model_kwargs["device_map"] = "auto"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            **model_kwargs
        )
        
        # 启用梯度检查点节省内存
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        logger.info(f"✅ 模型加载完成: {self.model.__class__.__name__}")
        logger.info(f"   - 参数量: {self.model.num_parameters() / 1e6:.1f}M")
    
    def setup_lora(self):
        """配置 LoRA 适配器"""
        if self.model is None:
            raise ValueError("请先调用 load_model() 加载模型")
        
        logger.info("🔧 配置 LoRA 适配器...")
        
        # 自动检测目标模块
        target_modules = self.config.target_modules
        if target_modules is None:
            # 根据模型类型自动选择
            model_type = self.model.config.model_type.lower()
            if "qwen" in model_type:
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
            elif "llama" in model_type or "mistral" in model_type:
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
            else:
                # 通用默认
                target_modules = ["q_proj", "v_proj"]
            logger.info(f"   - 自动检测目标模块: {target_modules}")
        
        # 创建 LoRA 配置
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        
        # 应用 LoRA
        self.peft_model = get_peft_model(self.model, lora_config)
        
        # 打印可训练参数
        trainable_params, all_params = self.peft_model.get_nb_trainable_parameters()
        logger.info(
            f"✅ LoRA 配置完成:\n"
            f"   - 可训练参数: {trainable_params / 1e6:.2f}M\n"
            f"   - 总参数: {all_params / 1e6:.2f}M\n"
            f"   - 可训练比例: {100 * trainable_params / all_params:.2f}%"
        )
    
    def load_data(
        self,
        data_path: str,
        eval_ratio: float = 0.1,
        data_format: str = "alpaca",
    ):
        """
        加载训练数据
        
        Args:
            data_path: 数据文件路径（JSON 或 JSONL）
            eval_ratio: 验证集比例
            data_format: 数据格式 (alpaca / openai / raw)
        """
        logger.info(f"📂 加载训练数据: {data_path}")
        
        # 读取数据
        if data_path.endswith(".jsonl"):
            with open(data_path, 'r', encoding='utf-8') as f:
                raw_data = [json.loads(line) for line in f if line.strip()]
        else:
            with open(data_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
        
        logger.info(f"   - 原始样本数: {len(raw_data)}")
        
        # 转换为统一格式
        processed_data = []
        for item in raw_data:
            text = self._format_training_example(item, data_format)
            if text:
                processed_data.append({"text": text})
        
        logger.info(f"   - 处理后样本数: {len(processed_data)}")
        
        # 创建数据集
        dataset = Dataset.from_list(processed_data)
        
        # 分割训练集和验证集
        if eval_ratio > 0:
            split = dataset.train_test_split(test_size=eval_ratio, seed=42)
            self.train_dataset = split["train"]
            self.eval_dataset = split["test"]
            logger.info(
                f"   - 训练集: {len(self.train_dataset)} 条\n"
                f"   - 验证集: {len(self.eval_dataset)} 条"
            )
        else:
            self.train_dataset = dataset
            self.eval_dataset = None
        
        # Tokenize 数据
        self._tokenize_dataset()
        
        logger.info("✅ 数据加载完成")
    
    def _format_training_example(
        self,
        item: Dict[str, Any],
        data_format: str,
    ) -> str:
        """将数据转换为训练格式"""
        
        if data_format == "alpaca":
            # Alpaca 格式: instruction, input, output
            instruction = item.get("instruction", "")
            input_text = item.get("input", "")
            output = item.get("output", "")
            
            if input_text:
                user_content = f"{instruction}\n\n参考信息：\n{input_text}"
            else:
                user_content = instruction
            
            messages = [
                {"role": "system", "content": "你是一个专业的问答助手，请准确、详细地回答用户问题。"},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": output},
            ]
            
        elif data_format == "openai":
            # OpenAI 格式: messages
            messages = item.get("messages", [])
            
        elif data_format == "raw":
            # 原始文本格式
            return item.get("text", "")
        
        else:
            raise ValueError(f"不支持的数据格式: {data_format}")
        
        # 使用分词器的 chat template
        if self.tokenizer is None:
            raise ValueError("请先调用 load_model() 加载模型")
        
        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            return text
        except Exception as e:
            logger.warning(f"格式化失败: {e}")
            return ""
    
    def _tokenize_dataset(self):
        """对数据集进行分词"""
        
        def tokenize_function(examples):
            # 分词
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.config.max_seq_length,
                padding=False,
            )
            # 对于因果语言模型，labels = input_ids
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized
        
        # 处理训练集
        self.train_dataset = self.train_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            desc="Tokenizing train set",
        )
        
        # 处理验证集
        if self.eval_dataset is not None:
            self.eval_dataset = self.eval_dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=["text"],
                desc="Tokenizing eval set",
            )
    
    def train(self):
        """
        执行 LoRA 微调训练
        """
        if self.peft_model is None:
            raise ValueError("请先调用 setup_lora() 配置 LoRA")
        if self.train_dataset is None:
            raise ValueError("请先调用 load_data() 加载数据")
        
        logger.info("🚀 开始训练...")
        
        # 创建输出目录
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            save_total_limit=3,
            eval_strategy="steps" if self.eval_dataset else "no",  # 新版 transformers 使用 eval_strategy
            eval_steps=self.config.save_steps if self.eval_dataset else None,
            load_best_model_at_end=True if self.eval_dataset else False,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            optim="adamw_torch",
            report_to="none",  # 禁用 wandb 等
            remove_unused_columns=False,
            dataloader_pin_memory=False if self.device == "cpu" else True,
            # CPU 特定优化
            use_cpu=True if self.device == "cpu" else False,
        )
        
        # 数据收集器
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            padding=True,
            return_tensors="pt",
        )
        
        # 创建 Trainer
        trainer = Trainer(
            model=self.peft_model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=data_collator,
        )
        
        # 开始训练
        logger.info(
            f"训练配置:\n"
            f"   - 轮数: {self.config.epochs}\n"
            f"   - 批次大小: {self.config.batch_size}\n"
            f"   - 梯度累积: {self.config.gradient_accumulation_steps}\n"
            f"   - 有效批次: {self.config.batch_size * self.config.gradient_accumulation_steps}\n"
            f"   - 学习率: {self.config.learning_rate}"
        )
        
        train_result = trainer.train()
        
        # 保存最终模型
        trainer.save_model()
        
        # 记录训练结果
        metrics = train_result.metrics
        logger.info(
            f"✅ 训练完成!\n"
            f"   - 总步数: {metrics.get('total_steps', 'N/A')}\n"
            f"   - 训练损失: {metrics.get('train_loss', 'N/A'):.4f}\n"
            f"   - 训练时间: {metrics.get('train_runtime', 0):.1f}s"
        )
        
        return metrics
    
    def save(self, output_path: Optional[str] = None):
        """
        保存微调后的模型
        
        Args:
            output_path: 输出路径，None 则使用配置中的路径
        """
        output_path = output_path or self.config.output_dir
        
        if self.peft_model is None:
            raise ValueError("没有可保存的模型")
        
        logger.info(f"💾 保存模型到: {output_path}")
        
        # 保存 LoRA 适配器
        adapter_path = os.path.join(output_path, "adapter")
        self.peft_model.save_pretrained(adapter_path)
        self.tokenizer.save_pretrained(adapter_path)
        
        logger.info(f"✅ LoRA 适配器已保存: {adapter_path}")
        
        # 保存配置
        config_path = os.path.join(output_path, "finetune_config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump({
                "base_model": self.config.base_model,
                "lora_rank": self.config.lora_rank,
                "lora_alpha": self.config.lora_alpha,
                "epochs": self.config.epochs,
                "max_seq_length": self.config.max_seq_length,
            }, f, indent=2, ensure_ascii=False)
        
        return adapter_path
    
    def merge_and_save(self, output_path: Optional[str] = None):
        """
        合并 LoRA 权重并保存完整模型
        
        Args:
            output_path: 输出路径
        """
        output_path = output_path or os.path.join(self.config.output_dir, "merged")
        
        if self.peft_model is None:
            raise ValueError("没有可合并的模型")
        
        logger.info("🔀 合并 LoRA 权重到基础模型...")
        
        # 合并权重
        merged_model = self.peft_model.merge_and_unload()
        
        # 保存合并后的模型
        merged_model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        
        logger.info(f"✅ 合并模型已保存: {output_path}")
        
        return output_path
    
    def load_adapter(self, adapter_path: str, base_model_path: Optional[str] = None):
        """
        加载已训练的 LoRA 适配器
        
        Args:
            adapter_path: 适配器路径
            base_model_path: 基础模型路径，None 则使用配置中的模型
        """
        base_model_path = base_model_path or self.config.base_model
        
        logger.info(f"📥 加载适配器: {adapter_path}")
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            adapter_path,
            trust_remote_code=True,
        )
        
        # 加载基础模型
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float32 if self.device == "cpu" else torch.float16,
        }
        
        if self.device == "cpu":
            model_kwargs["device_map"] = {"": "cpu"}
        else:
            model_kwargs["device_map"] = "auto"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            **model_kwargs
        )
        
        # 加载 LoRA 适配器
        self.peft_model = PeftModel.from_pretrained(
            self.model,
            adapter_path,
        )
        
        logger.info("✅ 适配器加载完成")
    
    def chat(
        self,
        message: str,
        system_prompt: str = "你是一个专业的问答助手，请准确、详细地回答用户问题。",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        do_sample: bool = True,
    ) -> str:
        """
        使用微调后的模型进行对话
        
        Args:
            message: 用户消息
            system_prompt: 系统提示词
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度
            do_sample: 是否使用采样
            
        Returns:
            str: 模型回复
        """
        model = self.peft_model if self.peft_model else self.model
        if model is None:
            raise ValueError("请先加载模型")
        
        # 构建对话
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
        ]
        
        # 应用 chat template
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # 编码
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        # 移动到正确设备
        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 生成
        model.eval()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        
        # 解码（只取新生成的部分）
        response = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        
        return response.strip()
    
    def run_full_pipeline(
        self,
        data_path: str = "./data/finetune/train_alpaca.json",
        test_prompt: str = "小米公司是什么时候成立的？创始人是谁？",
    ) -> Dict[str, Any]:
        """
        运行完整的微调流程
        
        Args:
            data_path: 训练数据路径
            test_prompt: 测试提示词
            
        Returns:
            Dict: 包含训练指标和测试结果
        """
        logger.info("=" * 60)
        logger.info("🎯 开始完整微调流程")
        logger.info("=" * 60)
        
        # Step 1: 加载模型
        logger.info("\n📥 Step 1: 加载基础模型")
        self.load_model()
        
        # Step 2: 配置 LoRA
        logger.info("\n🔧 Step 2: 配置 LoRA 适配器")
        self.setup_lora()
        
        # Step 3: 加载数据
        logger.info(f"\n📂 Step 3: 加载训练数据")
        self.load_data(data_path)
        
        # Step 4: 训练
        logger.info("\n🚀 Step 4: 开始训练")
        metrics = self.train()
        
        # Step 5: 保存
        logger.info("\n💾 Step 5: 保存模型")
        adapter_path = self.save()
        
        # Step 6: 测试
        logger.info("\n🧪 Step 6: 测试微调效果")
        response = self.chat(test_prompt)
        
        logger.info(f"\n📝 测试对话:")
        logger.info(f"   问: {test_prompt}")
        logger.info(f"   答: {response}")
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ 完整微调流程结束!")
        logger.info("=" * 60)
        
        return {
            "metrics": metrics,
            "adapter_path": adapter_path,
            "test_response": response,
        }


def quick_finetune(
    data_path: str = "./data/finetune/train_alpaca.json",
    model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    output_dir: str = "./models/finetuned_llm",
    epochs: int = 3,
    lora_rank: int = 8,
    device: str = "auto",
) -> Dict[str, Any]:
    """
    快速微调入口函数
    
    Args:
        data_path: 训练数据路径
        model: 基础模型
        output_dir: 输出目录
        epochs: 训练轮数
        lora_rank: LoRA 秩
        device: 训练设备
        
    Returns:
        Dict: 训练结果
        
    Example:
        >>> from src.stage_4.fine_tuning import quick_finetune
        >>> result = quick_finetune(
        ...     data_path="./data/finetune/train_alpaca.json",
        ...     model="Qwen/Qwen2.5-0.5B-Instruct",
        ...     epochs=3
        ... )
    """
    config = LocalFineTuneConfig(
        base_model=model,
        output_dir=output_dir,
        epochs=epochs,
        lora_rank=lora_rank,
        device=device,
    )
    
    finetuner = LocalLLMFineTuner(config)
    return finetuner.run_full_pipeline(data_path)

