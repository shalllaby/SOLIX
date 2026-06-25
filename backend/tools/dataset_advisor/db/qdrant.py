import logging
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from backend.tools.dataset_advisor.config import settings

logger = logging.getLogger("advisor.qdrant")

class QdrantManager:
    """Manages the lifecycle and operations of the Qdrant Vector database."""
    _client: QdrantClient = None
    COLLECTION_NAME = "datasets"

    def __init__(self):
        self.get_client()

    def get_client(self) -> QdrantClient:
        """Initialize and return Qdrant client based on configuration settings."""
        if self._client is None:
            url = settings.VECTOR_DB_URL
            
            if url == "local_storage":
                try:
                    logger.info(f"Initializing local persistent Qdrant Vector DB at path: {settings.VECTOR_DB_PATH}")
                    # Ensure path directory exists
                    os.makedirs(os.path.dirname(settings.VECTOR_DB_PATH), exist_ok=True)
                    self._client = QdrantClient(path=settings.VECTOR_DB_PATH)
                except Exception as ex:
                    logger.warning(f"Failed to initialize local persistent Qdrant (possibly locked by another process): {ex}. Falling back to in-memory volatile store.")
                    self._client = QdrantClient(location=":memory:")
            elif url == ":memory:":
                logger.info("Initializing in-memory volatile Qdrant Vector DB")
                self._client = QdrantClient(location=":memory:")
            else:
                logger.info(f"Connecting to remote Qdrant Server at: {url}")
                self._client = QdrantClient(url=url)
                
            self._ensure_collection_exists()
            
        return self._client

    def _ensure_collection_exists(self) -> None:
        """Create the datasets collection with standard model specifications if missing."""
        try:
            collections = self._client.get_collections().collections
            collection_names = [col.name for col in collections]
            
            if self.COLLECTION_NAME not in collection_names:
                logger.info(f"Creating vector collection '{self.COLLECTION_NAME}' (dim=384, Cosine similarity)...")
                self._client.create_collection(
                    collection_name=self.COLLECTION_NAME,
                    vectors_config=VectorParams(
                        size=384,  # Model output dimension of paraphrase-multilingual-MiniLM-L12-v2
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Collection '{self.COLLECTION_NAME}' created successfully.")
            else:
                logger.debug(f"Vector collection '{self.COLLECTION_NAME}' already exists.")
        except Exception as e:
            logger.error(f"Error ensuring Qdrant collection exists: {e}")

import os
# Global instantiator
qdrant_manager = QdrantManager()
qdrant_client = qdrant_manager.get_client()
