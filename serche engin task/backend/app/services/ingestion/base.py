from abc import ABC, abstractmethod
from typing import List, Optional
from pydantic import BaseModel, Field

class StandardizedDataset(BaseModel):
    """Unified schema representing a cleaned, standardized dataset format."""
    title: str = Field(..., description="Name/title of the dataset")
    description: str = Field("", description="Detailed description text")
    url: str = Field(..., description="Source URL link")
    kaggle_id: Optional[str] = Field(None, description="Kaggle unique identifier if applicable")
    provider: str = Field("kaggle", description="Identifier of the ingestion provider (e.g. kaggle, huggingface, openml)")
    row_count: Optional[int] = Field(None, description="Approximate number of rows")
    column_count: Optional[int] = Field(None, description="Approximate number of columns")
    license: Optional[str] = Field(None, description="Dataset usage license")
    task_type: Optional[str] = Field(None, description="Task classification: classification, regression, nlp, cv, time-series")
    language: Optional[str] = Field(None, description="Dataset primary language: arabic, english, multilingual")
    tags: List[str] = Field(default_factory=list, description="Descriptive tags/categories")
    quality_score: float = Field(0.0, description="Quality metric between 0.0 and 10.0")

class BaseDatasetProvider(ABC):
    """Abstract interface defining the requirements for ingestion scraping providers."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name identification of the dataset platform provider."""
        pass

    @abstractmethod
    async def fetch_datasets(self, query: str, limit: int = 20) -> List[StandardizedDataset]:
        """Fetch and clean datasets from source platform based on a category/search keyword."""
        pass
