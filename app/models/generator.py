import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Optional
from loguru import logger

from app.config import get_settings


class ThaiGenerator:
    """
    SeaLLM v3 model for Thai language generation.
    Optimized for Southeast Asian languages including Thai.
    """

    def __init__(self):
        self.settings = get_settings()
        self.model_name = self.settings.GENERATOR_MODEL
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info(f"Initializing {self.model_name} on {self.device}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            enable_thinking=False
        )

        # Load model with optimizations
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            attn_implementation="flash_attention_2",
        )

        if self.device == "cpu":
            self.model = self.model.to(self.device)

        self.model.eval()
        logger.info("Thai generation model loaded successfully")

    def create_rag_prompt(
        self,
        question: str,
        contexts: List[str],
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Create a RAG prompt from question and retrieved contexts.

        Args:
            question: User's question
            contexts: List of retrieved context passages
            system_prompt: Optional system prompt

        Returns:
            Formatted prompt string
        """
        if system_prompt is None:
            system_prompt = (
                "คุณเป็นผู้ช่วยที่เชี่ยวชาญในการตอบคำถามโดยอ้างอิงจากเอกสาร "
                "โปรดตอบคำถามโดยใช้ข้อมูลจากบริบทที่ให้มา "
                "ถ้าไม่พบข้อมูลในบริบท ให้บอกว่าไม่มีข้อมูลเพียงพอในการตอบ"
                "หากในบริบท มีเลขหน้า ให้ระบุเลขหน้าในคำตอบด้วย"
                "หากมีรูปภาพในบริบท ให้อ้างอิงรูปภาพในคำตอบด้วย"
            )

        # Combine contexts
        context_text = "\n\n".join([f"บริบทที่ {i+1}:\n{ctx}" for i, ctx in enumerate(contexts)])

        # Create prompt
        prompt = f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
บริบท:
{context_text}

คำถาม: {question}<|im_end|>
<|im_start|>assistant
"""

        return prompt

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = False
    ) -> str:
        """
        Generate text from prompt.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Repetition penalty
            do_sample: Whether to use sampling

        Returns:
            Generated text
        """
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        )

        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        # Decode
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        logger.debug(f"Generated {len(generated_text)} characters")
        return generated_text.strip()

    def generate_answer(
        self,
        question: str,
        contexts: List[str],
        max_new_tokens: int = 512,
        temperature: float = 0.7
    ) -> str:
        """
        Generate answer to question using retrieved contexts.

        Args:
            question: User's question
            contexts: Retrieved context passages
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Generated answer
        """
        if not contexts:
            return "ไม่พบข้อมูลที่เกี่ยวข้องในฐานข้อมูล กรุณาลองถามคำถามอื่นหรือเพิ่มเอกสารเข้าระบบ"

        # Create RAG prompt
        prompt = self.create_rag_prompt(question, contexts)

        # Generate answer
        answer = self.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=False
        )

        return answer

    def generate_summary(
        self,
        text: str,
        max_new_tokens: int = 256
    ) -> str:
        """
        Generate a summary of the given text.

        Args:
            text: Text to summarize
            max_new_tokens: Maximum tokens for summary

        Returns:
            Generated summary
        """
        prompt = f"""<|im_start|>system
คุณเป็นผู้ช่วยที่เชี่ยวชาญในการสรุปเอกสาร โปรดสรุปเนื้อหาที่กำหนดให้อย่างกระชับและชัดเจน<|im_end|>
<|im_start|>user
กรุณาสรุปเนื้อหาต่อไปนี้:

{text}<|im_end|>
<|im_start|>assistant
"""

        summary = self.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=0.5,
            do_sample=False
        )

        return summary


# Global instance
_generator_model: Optional[ThaiGenerator] = None


def get_generator_model() -> ThaiGenerator:
    """Get or create generator model instance"""
    global _generator_model
    if _generator_model is None:
        _generator_model = ThaiGenerator()
    return _generator_model
