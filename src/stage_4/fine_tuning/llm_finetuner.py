"""
LLM 微调数据准备器

自动生成高质量的 LLM 微调数据。
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import os
import json

from loguru import logger
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from src.stage_4.config import Stage4Config, get_stage4_config


class FineTuneDataFormat(str, Enum):
    """微调数据格式"""
    OPENAI = "openai"       # OpenAI 格式
    ALPACA = "alpaca"       # Alpaca 格式
    SHAREGPT = "sharegpt"   # ShareGPT 格式


@dataclass
class QAPair:
    """
    问答对
    
    用于 LLM 微调的数据格式。
    """
    question: str
    answer: str
    context: str = ""  # 可选的上下文
    difficulty: str = "medium"  # easy / medium / hard
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_openai_format(self, system_prompt: str = "") -> Dict[str, Any]:
        """转换为 OpenAI 微调格式"""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # 如果有上下文，添加到用户消息中
        user_content = self.question
        if self.context:
            user_content = f"参考信息：\n{self.context}\n\n问题：{self.question}"
        
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": self.answer})
        
        return {"messages": messages}
    
    def to_alpaca_format(self) -> Dict[str, str]:
        """转换为 Alpaca 格式"""
        return {
            "instruction": self.question,
            "input": self.context,
            "output": self.answer,
        }
    
    def to_sharegpt_format(self, system_prompt: str = "") -> Dict[str, Any]:
        """转换为 ShareGPT 格式"""
        conversations = []
        
        if system_prompt:
            conversations.append({"from": "system", "value": system_prompt})
        
        user_content = self.question
        if self.context:
            user_content = f"参考信息：\n{self.context}\n\n问题：{self.question}"
        
        conversations.append({"from": "human", "value": user_content})
        conversations.append({"from": "gpt", "value": self.answer})
        
        return {"conversations": conversations}
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "question": self.question,
            "answer": self.answer,
            "context": self.context,
            "difficulty": self.difficulty,
            "source": self.source,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QAPair":
        """从字典创建"""
        return cls(
            question=data["question"],
            answer=data["answer"],
            context=data.get("context", ""),
            difficulty=data.get("difficulty", "medium"),
            source=data.get("source", ""),
            metadata=data.get("metadata", {}),
        )


# QA 对生成提示词
QA_GENERATION_PROMPT = """基于以下文档内容，生成 {num_pairs} 个高质量的问答对。

文档内容：
{text}

要求：
1. 问题应该多样化，涵盖文档的不同方面
2. 答案应该准确、完整，可以直接从文档中找到依据
3. 难度级别：{difficulty}
   - easy: 简单的事实性问题，答案直接在文本中
   - medium: 需要简单理解和归纳的问题
   - hard: 需要综合分析多段信息的问题
4. 答案要详细，不要太简短

请按以下 JSON 格式输出：
{{
    "qa_pairs": [
        {{"question": "问题1", "answer": "详细答案1"}},
        {{"question": "问题2", "answer": "详细答案2"}}
    ]
}}"""


class LLMFineTuner:
    """
    LLM 微调数据准备器
    
    自动从文档生成高质量的 QA 对，用于 LLM 微调。
    """
    
    def __init__(
        self,
        config: Optional[Stage4Config] = None,
        output_dir: Optional[str] = None,
    ):
        """
        初始化 LLM 微调器
        
        Args:
            config: 配置
            output_dir: 输出目录
        """
        self.config = config or get_stage4_config()
        self.output_dir = output_dir or self.config.finetune_data_output_dir
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # QA 对存储
        self._qa_pairs: List[QAPair] = []
        
        # LLM
        self._llm = self._create_llm()
        
        logger.info(f"📝 LLM 微调器初始化完成: 输出目录={self.output_dir}")
    
    def _create_llm(self) -> ChatOpenAI:
        """创建 LLM 实例"""
        kwargs = {
            "model": self.config.model_name,
            "api_key": self.config.openai_api_key,
            "temperature": 0.8,  # 稍高的温度以增加多样性
        }
        if self.config.openai_base_url:
            kwargs["base_url"] = self.config.openai_base_url
        return ChatOpenAI(**kwargs)
    
    def _parse_qa_response(self, response: str) -> List[Dict[str, str]]:
        """解析 QA 生成响应"""
        try:
            content = response.strip()
            
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            
            data = json.loads(content)
            return data.get("qa_pairs", [])
            
        except json.JSONDecodeError as e:
            logger.warning(f"JSON 解析失败: {e}")
            return []
    
    def generate_qa_pairs(
        self,
        documents: List[Document],
        pairs_per_doc: int = None,
        difficulties: List[str] = None,
    ) -> List[QAPair]:
        """
        从文档生成 QA 对
        
        Args:
            documents: 文档列表
            pairs_per_doc: 每个文档生成的 QA 对数量
            difficulties: 难度级别列表
            
        Returns:
            List[QAPair]: 生成的 QA 对列表
        """
        pairs_per_doc = pairs_per_doc or self.config.qa_pairs_per_doc
        difficulties = difficulties or self.config.qa_difficulty_levels
        
        logger.info(f"🔄 生成 QA 对: {len(documents)} 个文档")
        
        all_pairs = []
        
        for i, doc in enumerate(documents):
            logger.info(f"处理文档 {i+1}/{len(documents)}")
            
            text = doc.page_content
            source = doc.metadata.get("file_name", f"doc_{i}")
            
            # 为每个难度级别生成 QA 对
            pairs_per_difficulty = max(1, pairs_per_doc // len(difficulties))
            
            for difficulty in difficulties:
                prompt = ChatPromptTemplate.from_template(QA_GENERATION_PROMPT)
                
                try:
                    response = self._llm.invoke(
                        prompt.format(
                            text=text[:3000],  # 限制长度
                            num_pairs=pairs_per_difficulty,
                            difficulty=difficulty,
                        )
                    )
                    
                    qa_data = self._parse_qa_response(response.content)
                    
                    for qa in qa_data:
                        pair = QAPair(
                            question=qa.get("question", ""),
                            answer=qa.get("answer", ""),
                            context=text[:500],  # 保留部分上下文
                            difficulty=difficulty,
                            source=source,
                        )
                        if pair.question and pair.answer:
                            all_pairs.append(pair)
                            
                except Exception as e:
                    logger.warning(f"QA 生成失败: {e}")
        
        self._qa_pairs.extend(all_pairs)
        
        logger.info(f"✅ 生成完成: {len(all_pairs)} 个 QA 对")
        
        return all_pairs
    
    def add_qa_pair(
        self,
        question: str,
        answer: str,
        context: str = "",
        difficulty: str = "medium",
        source: str = "",
    ):
        """手动添加 QA 对"""
        pair = QAPair(
            question=question,
            answer=answer,
            context=context,
            difficulty=difficulty,
            source=source,
        )
        self._qa_pairs.append(pair)
    
    def export_jsonl(
        self,
        filepath: Optional[str] = None,
        format: FineTuneDataFormat = FineTuneDataFormat.OPENAI,
        system_prompt: str = "你是一个专业的问答助手，请准确、详细地回答用户问题。",
    ):
        """
        导出为 JSONL 格式
        
        Args:
            filepath: 输出文件路径
            format: 数据格式
            system_prompt: 系统提示词
        """
        filepath = filepath or os.path.join(
            self.output_dir,
            f"train_{format.value}.jsonl"
        )
        
        with open(filepath, 'w', encoding='utf-8') as f:
            for pair in self._qa_pairs:
                if format == FineTuneDataFormat.OPENAI:
                    data = pair.to_openai_format(system_prompt)
                elif format == FineTuneDataFormat.ALPACA:
                    data = pair.to_alpaca_format()
                elif format == FineTuneDataFormat.SHAREGPT:
                    data = pair.to_sharegpt_format(system_prompt)
                else:
                    data = pair.to_dict()
                
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
        
        logger.info(f"数据已导出: {filepath} ({len(self._qa_pairs)} 条)")
    
    def export_json(
        self,
        filepath: Optional[str] = None,
        format: FineTuneDataFormat = FineTuneDataFormat.ALPACA,
        system_prompt: str = "",
    ):
        """
        导出为 JSON 格式
        
        Args:
            filepath: 输出文件路径
            format: 数据格式
            system_prompt: 系统提示词
        """
        filepath = filepath or os.path.join(
            self.output_dir,
            f"train_{format.value}.json"
        )
        
        data = []
        for pair in self._qa_pairs:
            if format == FineTuneDataFormat.OPENAI:
                data.append(pair.to_openai_format(system_prompt))
            elif format == FineTuneDataFormat.ALPACA:
                data.append(pair.to_alpaca_format())
            elif format == FineTuneDataFormat.SHAREGPT:
                data.append(pair.to_sharegpt_format(system_prompt))
            else:
                data.append(pair.to_dict())
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"数据已导出: {filepath} ({len(self._qa_pairs)} 条)")
    
    def save_qa_pairs(self, filepath: Optional[str] = None):
        """保存 QA 对（原始格式）"""
        filepath = filepath or os.path.join(self.output_dir, "qa_pairs.json")
        
        data = [pair.to_dict() for pair in self._qa_pairs]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"QA 对已保存: {filepath}")
    
    def load_qa_pairs(self, filepath: str):
        """加载 QA 对"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self._qa_pairs = [QAPair.from_dict(item) for item in data]
        
        logger.info(f"QA 对已加载: {len(self._qa_pairs)} 条")
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        difficulty_counts = {}
        source_counts = {}
        
        for pair in self._qa_pairs:
            difficulty_counts[pair.difficulty] = difficulty_counts.get(pair.difficulty, 0) + 1
            source_counts[pair.source] = source_counts.get(pair.source, 0) + 1
        
        return {
            "total_pairs": len(self._qa_pairs),
            "difficulty_distribution": difficulty_counts,
            "source_distribution": source_counts,
        }
    
    def filter_by_difficulty(self, difficulty: str) -> List[QAPair]:
        """按难度筛选"""
        return [p for p in self._qa_pairs if p.difficulty == difficulty]
    
    def split_train_test(
        self,
        test_ratio: float = 0.1,
    ) -> tuple[List[QAPair], List[QAPair]]:
        """
        划分训练集和测试集
        
        Args:
            test_ratio: 测试集比例
            
        Returns:
            Tuple: (训练集, 测试集)
        """
        import random
        
        pairs = self._qa_pairs.copy()
        random.shuffle(pairs)
        
        split_idx = int(len(pairs) * (1 - test_ratio))
        
        return pairs[:split_idx], pairs[split_idx:]
    
    @property
    def num_qa_pairs(self) -> int:
        """QA 对数量"""
        return len(self._qa_pairs)

