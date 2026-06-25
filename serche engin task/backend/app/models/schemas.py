from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class SearchRequest(BaseModel):
    """Input parameters for a conversational search request."""
    query: str = Field(..., description="The natural language dataset query", min_length=2)
    session_id: str = Field("default-session", description="Session identifier for stateful chats")

class QueryAnalysis(BaseModel):
    """Extract filters parsed from the search intent."""
    task_type: Optional[str] = None
    modality: Optional[str] = None
    language: Optional[str] = None
    max_rows: Optional[int] = None
    min_rows: Optional[int] = None
    country: Optional[str] = None
    semantic_search_query: str
    detected_language: str

class DatasetCardResponse(BaseModel):
    """Output schema representing an individual dataset match with AI insights."""
    id: str
    title: str
    description: str
    url: str
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    license: Optional[str] = None
    task_type: Optional[str] = None
    language: Optional[str] = None
    tags: List[str] = []
    relevance_score: float
    reasoning: str

class SearchResponse(BaseModel):
    """Top-level response payload for a conversational query."""
    query_analysis: QueryAnalysis
    detected_language: str
    advisor_message: str
    datasets: List[DatasetCardResponse]

class StatsResponse(BaseModel):
    """Database statistics overview."""
    datasets_count: int
    search_logs_count: int
    recommendations_count: int
    qdrant_status: str
    engine_dialect: str
