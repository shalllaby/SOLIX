import logging
from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer
from backend.app.config import settings

logger = logging.getLogger("advisor.embeddings")

class EmbeddingService:
    _instance = None
    _model = None

    def __new__(cls, *args, **kwargs):
        """Singleton pattern to avoid multiple loads of the embedding model in memory."""
        if cls._instance is None:
            cls._instance = super(EmbeddingService, cls).__new__(cls)
        return cls._instance

    @property
    def model(self) -> SentenceTransformer:
        """Lazy-loaded local sentence transformer model."""
        if self._model is None:
            logger.info(f"Loading local embedding model: {settings.EMBEDDING_MODEL_NAME}...")
            try:
                self._model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
                logger.info("Local embedding model loaded successfully.")
            except Exception as e:
                logger.error(f"Error loading embedding model: {e}")
                raise e
        return self._model

    def get_embedding(self, text: str) -> List[float]:
        """Convert a single text string into a 384-dimensional dense vector."""
        if not text.strip():
            return [0.0] * 384
        
        # sentence-transformers encodes text directly to a numpy array
        vector = self.model.encode(text, convert_to_numpy=True)
        return vector.tolist()

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Batch vectorize multiple text segments efficiently."""
        if not texts:
            return []
        
        vectors = self.model.encode(texts, convert_to_numpy=True, batch_size=32, show_progress_bar=False)
        return vectors.tolist()

# Global instantiator for easy reuse across app
embedding_service = EmbeddingService()
