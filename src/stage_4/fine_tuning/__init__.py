"""
微调模块

提供 Embedding 和 LLM 微调能力：
- Embedding 模型微调
- LLM 微调数据准备
- 训练数据生成
- 本地 LLM LoRA 微调
"""

from .embedding_finetuner import (
    EmbeddingFineTuner,
    TrainingPair,
    TrainingTriplet,
)
from .llm_finetuner import (
    LLMFineTuner,
    QAPair,
    FineTuneDataFormat,
)
from .training_data_generator import (
    TrainingDataGenerator,
)
from .local_llm_finetuner import (
    LocalLLMFineTuner,
    LocalFineTuneConfig,
    quick_finetune,
)

__all__ = [
    # Embedding 微调
    "EmbeddingFineTuner",
    "TrainingPair",
    "TrainingTriplet",
    # LLM 微调数据生成
    "LLMFineTuner",
    "QAPair",
    "FineTuneDataFormat",
    # 数据生成
    "TrainingDataGenerator",
    # 本地 LLM 微调
    "LocalLLMFineTuner",
    "LocalFineTuneConfig",
    "quick_finetune",
]

