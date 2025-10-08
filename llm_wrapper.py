# backend/llm_wrapper.py
from typing import Optional
import logging
logger = logging.getLogger(__name__)

class LocalLLM:
    def __init__(self, device: Optional[int]=None, use_fallback: bool=False):
        self.use_fallback = use_fallback
        self.pipeline = None
        self.device = device
        if not use_fallback:
            try:
                from transformers import pipeline as hf_pipeline
                import torch
                if device is None:
                    device = 0 if torch.cuda.is_available() else -1
                self.device = device
                self.pipeline = hf_pipeline('sentiment-analysis', model='distilbert-base-uncased', device=self.device)
                logger.info('Loaded HF local sentiment pipeline.')
            except Exception as e:
                logger.warning(f'Could not load HF pipeline: {e}; falling back.')
                self.pipeline = None
                self.use_fallback = True

    def generate_sentiment(self, text: str) -> float:
        if self.pipeline is not None:
            try:
                out = self.pipeline(text[:512])
                label = out[0].get('label','').upper()
                score = float(out[0].get('score',0.0))
                return score if label.startswith('POS') else -score
            except Exception as e:
                logger.warning(f'Error calling pipeline: {e}; switching to fallback.')
                self.pipeline = None
                self.use_fallback = True
        txt = (text or '').lower()
        pos = sum(1 for k in ['up','gain','rise','surge','beat','optim','upgrade','bull'] if k in txt)
        neg = sum(1 for k in ['down','drop','fall','loss','miss','downgrade','bear','sell','weak'] if k in txt)
        score = 0.2*pos - 0.2*neg
        if score > 1: score = 1.0
        if score < -1: score = -1.0
        return float(score)
