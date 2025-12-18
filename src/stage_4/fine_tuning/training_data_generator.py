"""
训练数据生成器

综合多种方法生成高质量训练数据。
"""

from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
import os
import json
import random

from loguru import logger
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.stage_4.config import Stage4Config, get_stage4_config
from .embedding_finetuner import TrainingPair, TrainingTriplet
from .llm_finetuner import QAPair


# 数据增强提示词
PARAPHRASE_PROMPT = """请将以下问题改写成不同的表达方式，保持意思不变。
生成 {num_paraphrases} 个不同的改写版本。

原问题：{question}

请按以下 JSON 格式输出：
{{"paraphrases": ["改写1", "改写2", ...]}}"""

# 难题生成提示词
HARD_QUESTION_PROMPT = """基于以下多段文档内容，生成需要综合分析才能回答的问题。

文档内容：
{texts}

要求：
1. 问题应该需要结合多段文档的信息才能回答
2. 问题应该有一定难度，不是简单的事实查找
3. 生成 {num_questions} 个这样的问题，并给出答案

请按以下 JSON 格式输出：
{{
    "questions": [
        {{"question": "问题1", "answer": "答案1"}},
        {{"question": "问题2", "answer": "答案2"}}
    ]
}}"""


class TrainingDataGenerator:
    """
    训练数据生成器
    
    综合多种方法生成高质量训练数据：
    - 问答对生成
    - 数据增强（改写）
    - 负样本挖掘
    - 难题生成
    """
    
    def __init__(
        self,
        config: Optional[Stage4Config] = None,
        output_dir: Optional[str] = None,
    ):
        """
        初始化训练数据生成器
        
        Args:
            config: 配置
            output_dir: 输出目录
        """
        self.config = config or get_stage4_config()
        self.output_dir = output_dir or os.path.join(
            self.config.finetune_data_output_dir,
            "generated"
        )
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # LLM
        self._llm = self._create_llm()
        
        # 数据存储
        self._qa_pairs: List[QAPair] = []
        self._embedding_pairs: List[TrainingPair] = []
        self._embedding_triplets: List[TrainingTriplet] = []
        
        logger.info(f"🔧 训练数据生成器初始化完成")
    
    def _create_llm(self) -> ChatOpenAI:
        """创建 LLM 实例"""
        kwargs = {
            "model": self.config.model_name,
            "api_key": self.config.openai_api_key,
            "temperature": 0.8,
        }
        if self.config.openai_base_url:
            kwargs["base_url"] = self.config.openai_base_url
        return ChatOpenAI(**kwargs)
    
    def generate_paraphrases(
        self,
        question: str,
        num_paraphrases: int = 3,
    ) -> List[str]:
        """
        生成问题的改写版本
        
        Args:
            question: 原始问题
            num_paraphrases: 改写数量
            
        Returns:
            List[str]: 改写版本列表
        """
        try:
            prompt = ChatPromptTemplate.from_template(PARAPHRASE_PROMPT)
            response = self._llm.invoke(
                prompt.format(question=question, num_paraphrases=num_paraphrases)
            )
            
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            data = json.loads(content)
            return data.get("paraphrases", [])
            
        except Exception as e:
            logger.warning(f"改写生成失败: {e}")
            return []
    
    def generate_hard_questions(
        self,
        documents: List[Document],
        num_questions: int = 5,
    ) -> List[QAPair]:
        """
        生成需要综合分析的难题
        
        Args:
            documents: 文档列表
            num_questions: 生成数量
            
        Returns:
            List[QAPair]: 问答对列表
        """
        if len(documents) < 2:
            logger.warning("需要至少 2 个文档才能生成综合性问题")
            return []
        
        # 随机选择几个文档组合
        selected_docs = random.sample(documents, min(len(documents), 3))
        texts = "\n\n---\n\n".join([
            f"[文档 {i+1}]\n{doc.page_content[:1000]}"
            for i, doc in enumerate(selected_docs)
        ])
        
        try:
            prompt = ChatPromptTemplate.from_template(HARD_QUESTION_PROMPT)
            response = self._llm.invoke(
                prompt.format(texts=texts, num_questions=num_questions)
            )
            
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            data = json.loads(content)
            
            pairs = []
            for qa in data.get("questions", []):
                pair = QAPair(
                    question=qa.get("question", ""),
                    answer=qa.get("answer", ""),
                    context=texts[:500],
                    difficulty="hard",
                    source="multi_doc",
                )
                if pair.question and pair.answer:
                    pairs.append(pair)
            
            return pairs
            
        except Exception as e:
            logger.warning(f"难题生成失败: {e}")
            return []
    
    def augment_qa_pairs(
        self,
        qa_pairs: List[QAPair],
        num_augments: int = 2,
    ) -> List[QAPair]:
        """
        通过改写增强 QA 对
        
        Args:
            qa_pairs: 原始 QA 对
            num_augments: 每个 QA 对的增强数量
            
        Returns:
            List[QAPair]: 增强后的 QA 对列表
        """
        augmented = []
        
        for i, pair in enumerate(qa_pairs):
            logger.info(f"增强 QA 对 {i+1}/{len(qa_pairs)}")
            
            # 原始数据
            augmented.append(pair)
            
            # 生成改写
            paraphrases = self.generate_paraphrases(pair.question, num_augments)
            
            for paraphrase in paraphrases:
                new_pair = QAPair(
                    question=paraphrase,
                    answer=pair.answer,
                    context=pair.context,
                    difficulty=pair.difficulty,
                    source=pair.source,
                    metadata={"augmented_from": pair.question},
                )
                augmented.append(new_pair)
        
        logger.info(f"✅ 增强完成: {len(qa_pairs)} -> {len(augmented)}")
        return augmented
    
    def generate_hard_negatives(
        self,
        documents: List[Document],
        qa_pairs: List[QAPair],
    ) -> List[TrainingTriplet]:
        """
        为 Embedding 训练生成困难负样本
        
        困难负样本：与正样本相似但实际上不相关的样本
        
        Args:
            documents: 文档列表
            qa_pairs: QA 对列表
            
        Returns:
            List[TrainingTriplet]: 训练三元组列表
        """
        triplets = []
        all_texts = [doc.page_content for doc in documents]
        
        for pair in qa_pairs:
            # 找到与正样本不同但可能混淆的负样本
            positives = pair.context or pair.answer
            
            # 选择主题相似但内容不同的负样本
            negative_candidates = [
                t for t in all_texts
                if t != positives and len(t) > 100
            ]
            
            if negative_candidates:
                # 随机选择一个作为困难负样本
                negative = random.choice(negative_candidates)
                
                triplet = TrainingTriplet(
                    anchor=pair.question,
                    positive=positives[:500],
                    negative=negative[:500],
                )
                triplets.append(triplet)
        
        logger.info(f"生成了 {len(triplets)} 个困难负样本三元组")
        return triplets
    
    def generate_all(
        self,
        documents: List[Document],
        qa_pairs_per_doc: int = 5,
        augment: bool = True,
        generate_hard: bool = True,
    ) -> Dict[str, Any]:
        """
        一站式生成所有训练数据
        
        Args:
            documents: 文档列表
            qa_pairs_per_doc: 每个文档的 QA 对数量
            augment: 是否进行数据增强
            generate_hard: 是否生成难题
            
        Returns:
            Dict: 包含所有生成数据的字典
        """
        logger.info(f"🚀 开始一站式数据生成: {len(documents)} 个文档")
        
        from .llm_finetuner import LLMFineTuner
        from .embedding_finetuner import EmbeddingFineTuner
        
        # 1. 基础 QA 对生成
        llm_finetuner = LLMFineTuner(config=self.config)
        qa_pairs = llm_finetuner.generate_qa_pairs(
            documents,
            pairs_per_doc=qa_pairs_per_doc,
        )
        
        # 2. 数据增强
        if augment:
            qa_pairs = self.augment_qa_pairs(qa_pairs, num_augments=2)
        
        # 3. 难题生成
        if generate_hard and len(documents) >= 2:
            hard_pairs = self.generate_hard_questions(documents, num_questions=5)
            qa_pairs.extend(hard_pairs)
        
        # 4. Embedding 训练数据
        embedding_finetuner = EmbeddingFineTuner(config=self.config)
        emb_pairs, emb_triplets = embedding_finetuner.generate_training_data(documents)
        
        # 5. 困难负样本
        hard_triplets = self.generate_hard_negatives(documents, qa_pairs)
        emb_triplets.extend(hard_triplets)
        
        # 存储
        self._qa_pairs = qa_pairs
        self._embedding_pairs = emb_pairs
        self._embedding_triplets = emb_triplets
        
        result = {
            "qa_pairs": qa_pairs,
            "embedding_pairs": emb_pairs,
            "embedding_triplets": emb_triplets,
            "statistics": {
                "total_qa_pairs": len(qa_pairs),
                "total_embedding_pairs": len(emb_pairs),
                "total_embedding_triplets": len(emb_triplets),
            }
        }
        
        logger.info(
            f"✅ 一站式数据生成完成:\n"
            f"   - QA 对: {len(qa_pairs)}\n"
            f"   - Embedding 训练对: {len(emb_pairs)}\n"
            f"   - Embedding 三元组: {len(emb_triplets)}"
        )
        
        return result
    
    def save_all(self, prefix: str = ""):
        """保存所有生成的数据"""
        prefix = prefix or "generated"
        
        # 保存 QA 对
        qa_path = os.path.join(self.output_dir, f"{prefix}_qa_pairs.json")
        with open(qa_path, 'w', encoding='utf-8') as f:
            json.dump(
                [p.to_dict() for p in self._qa_pairs],
                f,
                ensure_ascii=False,
                indent=2
            )
        
        # 保存 Embedding 训练数据
        emb_path = os.path.join(self.output_dir, f"{prefix}_embedding_data.json")
        with open(emb_path, 'w', encoding='utf-8') as f:
            json.dump(
                {
                    "pairs": [p.to_dict() for p in self._embedding_pairs],
                    "triplets": [t.to_dict() for t in self._embedding_triplets],
                },
                f,
                ensure_ascii=False,
                indent=2
            )
        
        logger.info(f"所有数据已保存到: {self.output_dir}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        difficulty_dist = {}
        for p in self._qa_pairs:
            difficulty_dist[p.difficulty] = difficulty_dist.get(p.difficulty, 0) + 1
        
        return {
            "qa_pairs": {
                "total": len(self._qa_pairs),
                "difficulty_distribution": difficulty_dist,
            },
            "embedding_data": {
                "pairs": len(self._embedding_pairs),
                "triplets": len(self._embedding_triplets),
            }
        }

