"""
微调模块

提供 Embedding 和 LLM 微调能力：
- Embedding 模型微调
- LLM 微调数据准备
- 训练数据生成
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

__all__ = [
    # Embedding 微调
    "EmbeddingFineTuner",
    "TrainingPair",
    "TrainingTriplet",
    # LLM 微调
    "LLMFineTuner",
    "QAPair",
    "FineTuneDataFormat",
    # 数据生成
    "TrainingDataGenerator",
]

