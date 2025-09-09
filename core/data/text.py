import torch
from transformers import AutoTokenizer
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class TextProcessor:
    """Text processing for lyrics embeddings."""
    
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        max_length: int = 512,
        padding: str = "max_length",
        truncation: bool = True
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add special tokens if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text (clean, normalize, etc.)."""
        # Basic text cleaning
        text = text.strip()
        
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        # Remove common artifacts
        text = text.replace("[Instrumental]", "")
        text = text.replace("[Chorus]", "")
        text = text.replace("[Verse]", "")
        text = text.replace("[Bridge]", "")
        text = text.replace("[Outro]", "")
        
        return text
    
    def tokenize(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize text and return input tensors."""
        # Preprocess text
        text = self.preprocess_text(text)
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0)
        }
    
    def batch_tokenize(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize multiple texts in batch."""
        # Preprocess all texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Tokenize batch
        encoding = self.tokenizer(
            processed_texts,
            max_length=self.max_length,
            padding=self.padding,
            truncation=self.truncation,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"]
        }
    
    def decode(self, input_ids: torch.Tensor) -> str:
        """Decode token IDs back to text."""
        return self.tokenizer.decode(input_ids, skip_special_tokens=True)
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return self.tokenizer.vocab_size
    
    def get_special_tokens(self) -> Dict[str, int]:
        """Get special token IDs."""
        return {
            "pad_token_id": self.tokenizer.pad_token_id,
            "unk_token_id": self.tokenizer.unk_token_id,
            "cls_token_id": self.tokenizer.cls_token_id,
            "sep_token_id": self.tokenizer.sep_token_id,
            "mask_token_id": self.tokenizer.mask_token_id
        }


def create_text_processor(config) -> TextProcessor:
    """Create text processor from config."""
    return TextProcessor(
        model_name=config.hf_text_model,
        max_length=config.max_text_length
    )
