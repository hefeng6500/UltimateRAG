"""
Embedding 微调器

使用 Sentence Transformers 微调 Embedding 模型。
"""

from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
import os
import json
import random

from loguru import logger
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.stage_4.config import Stage4Config, get_stage4_config


@dataclass
class TrainingPair:
    """
    训练对（正样本对）
    
    用于对比学习：anchor 和 positive 应该相似
    """
    anchor: str      # 锚点文本（如：问题）
    positive: str    # 正样本（如：相关答案）
    
    def to_dict(self) -> Dict[str, str]:
        return {"anchor": self.anchor, "positive": self.positive}


@dataclass
class TrainingTriplet:
    """
    训练三元组
    
    用于对比学习：
    - anchor 和 positive 应该相似
    - anchor 和 negative 应该不相似
    """
    anchor: str      # 锚点文本
    positive: str    # 正样本
    negative: str    # 负样本
    
    def to_dict(self) -> Dict[str, str]:
        return {
            "anchor": self.anchor,
            "positive": self.positive,
            "negative": self.negative,
        }


# 问题生成提示词
QUESTION_GENERATION_PROMPT = """基于以下文本内容，生成 {num_questions} 个可以用这段文本回答的问题。

文本内容：
{text}

要求：
1. 问题应该多样化，覆盖文本的不同方面
2. 问题应该具体，不要太宽泛
3. 确保问题可以用给定文本回答

请按以下 JSON 格式输出：
{{"questions": ["问题1", "问题2", ...]}}"""


class EmbeddingFineTuner:
    """
    Embedding 微调器
    
    使用对比学习微调 Embedding 模型，使其更适应特定领域。
    """
    
    def __init__(
        self,
        base_model: Optional[str] = None,
        output_dir: Optional[str] = None,
        config: Optional[Stage4Config] = None,
    ):
        """
        初始化 Embedding 微调器
        
        Args:
            base_model: 基础模型名称
            output_dir: 输出目录
            config: 配置
        """
        self.config = config or get_stage4_config()
        self.base_model = base_model or self.config.embedding_finetune_model
        self.output_dir = output_dir or self.config.embedding_finetune_output_dir
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 训练数据
        self._training_pairs: List[TrainingPair] = []
        self._training_triplets: List[TrainingTriplet] = []
        
        # LLM 用于生成问题
        self._llm = self._create_llm()
        
        logger.info(f"📚 Embedding 微调器初始化完成: 基础模型={self.base_model}")
    
    def _create_llm(self) -> ChatOpenAI:
        """创建 LLM 实例"""
        kwargs = {
            "model": self.config.model_name,
            "api_key": self.config.openai_api_key,
            "temperature": 0.7,
        }
        if self.config.openai_base_url:
            kwargs["base_url"] = self.config.openai_base_url
        return ChatOpenAI(**kwargs)
    
    def _generate_questions(self, text: str, num_questions: int = 3) -> List[str]:
        """从文本生成问题"""
        try:
            prompt = ChatPromptTemplate.from_template(QUESTION_GENERATION_PROMPT)
            response = self._llm.invoke(
                prompt.format(text=text[:2000], num_questions=num_questions)
            )
            
            content = response.content.strip()
            
            # 解析 JSON
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            data = json.loads(content)
            return data.get("questions", [])
            
        except Exception as e:
            logger.warning(f"问题生成失败: {e}")
            return []
    
    def generate_training_data(
        self,
        documents: List[Document],
        questions_per_doc: int = 3,
        include_triplets: bool = True,
    ) -> Tuple[List[TrainingPair], List[TrainingTriplet]]:
        """
        从文档生成训练数据
        
        Args:
            documents: 文档列表
            questions_per_doc: 每个文档生成的问题数
            include_triplets: 是否生成三元组
            
        Returns:
            Tuple: (训练对列表, 训练三元组列表)
        """
        logger.info(f"🔄 生成训练数据: {len(documents)} 个文档")
        
        pairs = []
        triplets = []
        all_texts = [doc.page_content for doc in documents]
        
        for i, doc in enumerate(documents):
            logger.info(f"处理文档 {i+1}/{len(documents)}")
            
            text = doc.page_content
            
            # 生成问题
            questions = self._generate_questions(text, questions_per_doc)
            
            for question in questions:
                # 创建正样本对
                pair = TrainingPair(anchor=question, positive=text)
                pairs.append(pair)
                
                # 创建三元组（添加负样本）
                if include_triplets and len(all_texts) > 1:
                    # 随机选择一个不同的文档作为负样本
                    negative_texts = [t for t in all_texts if t != text]
                    if negative_texts:
                        negative = random.choice(negative_texts)
                        triplet = TrainingTriplet(
                            anchor=question,
                            positive=text,
                            negative=negative,
                        )
                        triplets.append(triplet)
        
        self._training_pairs.extend(pairs)
        self._training_triplets.extend(triplets)
        
        logger.info(f"✅ 生成完成: {len(pairs)} 个训练对, {len(triplets)} 个三元组")
        
        return pairs, triplets
    
    def add_training_pair(self, anchor: str, positive: str):
        """手动添加训练对"""
        self._training_pairs.append(TrainingPair(anchor=anchor, positive=positive))
    
    def add_training_triplet(self, anchor: str, positive: str, negative: str):
        """手动添加训练三元组"""
        self._training_triplets.append(
            TrainingTriplet(anchor=anchor, positive=positive, negative=negative)
        )
    
    def save_training_data(self, filepath: Optional[str] = None):
        """保存训练数据"""
        filepath = filepath or os.path.join(self.output_dir, "training_data.json")
        
        data = {
            "pairs": [p.to_dict() for p in self._training_pairs],
            "triplets": [t.to_dict() for t in self._training_triplets],
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"训练数据已保存: {filepath}")
    
    def load_training_data(self, filepath: str):
        """加载训练数据"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self._training_pairs = [
            TrainingPair(**p) for p in data.get("pairs", [])
        ]
        self._training_triplets = [
            TrainingTriplet(**t) for t in data.get("triplets", [])
        ]
        
        logger.info(
            f"训练数据已加载: {len(self._training_pairs)} 个训练对, "
            f"{len(self._training_triplets)} 个三元组"
        )
    
    def train(
        self,
        epochs: int = None,
        batch_size: int = None,
        learning_rate: float = None,
        use_triplets: bool = True,
    ):
        """
        执行微调训练
        
        Args:
            epochs: 训练轮数
            batch_size: 批次大小
            learning_rate: 学习率
            use_triplets: 是否使用三元组训练
        """
        epochs = epochs or self.config.embedding_finetune_epochs
        batch_size = batch_size or self.config.embedding_finetune_batch_size
        learning_rate = learning_rate or self.config.embedding_finetune_lr
        
        try:
            from sentence_transformers import SentenceTransformer, InputExample, losses
            from torch.utils.data import DataLoader
        except ImportError:
            logger.error("请安装 sentence-transformers: pip install sentence-transformers")
            return
        
        logger.info(f"🚀 开始微调训练: {self.base_model}")
        logger.info(f"   - 轮数: {epochs}")
        logger.info(f"   - 批次大小: {batch_size}")
        logger.info(f"   - 学习率: {learning_rate}")
        
        # 加载基础模型
        model = SentenceTransformer(self.base_model)
        
        # 准备训练数据
        train_examples = []
        
        if use_triplets and self._training_triplets:
            # 使用三元组
            for triplet in self._training_triplets:
                train_examples.append(InputExample(
                    texts=[triplet.anchor, triplet.positive, triplet.negative]
                ))
            
            train_dataloader = DataLoader(
                train_examples,
                shuffle=True,
                batch_size=batch_size,
            )
            
            # 使用 TripletLoss
            train_loss = losses.TripletLoss(model=model)
            
        else:
            # 使用对比对
            for pair in self._training_pairs:
                train_examples.append(InputExample(
                    texts=[pair.anchor, pair.positive],
                    label=1.0,  # 相似度分数
                ))
            
            train_dataloader = DataLoader(
                train_examples,
                shuffle=True,
                batch_size=batch_size,
            )
            
            # 使用 CosineSimilarityLoss
            train_loss = losses.CosineSimilarityLoss(model=model)
        
        # 训练
        warmup_steps = int(len(train_dataloader) * epochs * 0.1)
        
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=epochs,
            warmup_steps=warmup_steps,
            output_path=self.output_dir,
            show_progress_bar=True,
        )
        
        logger.info(f"✅ 微调完成，模型已保存: {self.output_dir}")
    
    def evaluate(self, test_pairs: List[TrainingPair]) -> Dict[str, float]:
        """
        评估微调后的模型
        
        Args:
            test_pairs: 测试数据对
            
        Returns:
            Dict: 评估指标
        """
        try:
            from sentence_transformers import SentenceTransformer
            from sklearn.metrics.pairwise import cosine_similarity
            import numpy as np
        except ImportError:
            logger.error("请安装必要的包")
            return {}
        
        # 加载微调后的模型
        model = SentenceTransformer(self.output_dir)
        
        # 计算相似度
        similarities = []
        for pair in test_pairs:
            anchor_emb = model.encode([pair.anchor])
            positive_emb = model.encode([pair.positive])
            sim = cosine_similarity(anchor_emb, positive_emb)[0][0]
            similarities.append(sim)
        
        return {
            "mean_similarity": float(np.mean(similarities)),
            "std_similarity": float(np.std(similarities)),
            "min_similarity": float(np.min(similarities)),
            "max_similarity": float(np.max(similarities)),
        }
    
    def get_model_path(self) -> str:
        """获取微调后的模型路径"""
        return self.output_dir
    
    @property
    def num_training_pairs(self) -> int:
        """训练对数量"""
        return len(self._training_pairs)
    
    @property
    def num_training_triplets(self) -> int:
        """训练三元组数量"""
        return len(self._training_triplets)

