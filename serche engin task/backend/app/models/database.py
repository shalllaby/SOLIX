from datetime import datetime
import uuid
from typing import List, Optional, Dict, Any
from sqlalchemy import String, Text, Float, Integer, DateTime, JSON, ForeignKey, Table, Column
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

class Base(DeclarativeBase):
    pass

class Dataset(Base):
    __tablename__ = "datasets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    kaggle_id: Mapped[Optional[str]] = mapped_column(String(255), unique=True, nullable=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    url: Mapped[str] = mapped_column(String(512), nullable=False)
    row_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    column_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    license: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)
    task_type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True) # e.g., classification, regression, nlp
    language: Mapped[Optional[str]] = mapped_column(String(50), nullable=True) # e.g., arabic, english, multilingual
    tags: Mapped[Optional[List[str]]] = mapped_column(JSON, nullable=True) # Cross-compatible JSON list
    quality_score: Mapped[float] = mapped_column(Float, default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    recommendations: Mapped[List["Recommendation"]] = relationship("Recommendation", back_populates="dataset", cascade="all, delete-orphan")

class SearchLog(Base):
    __tablename__ = "search_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    session_id: Mapped[str] = mapped_column(String(255), nullable=False)
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    detected_language: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    extracted_filters: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    recommendations: Mapped[List["Recommendation"]] = relationship("Recommendation", back_populates="search_log", cascade="all, delete-orphan")

class Recommendation(Base):
    __tablename__ = "recommendations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    log_id: Mapped[str] = mapped_column(String(36), ForeignKey("search_logs.id", ondelete="CASCADE"), nullable=False)
    dataset_id: Mapped[str] = mapped_column(String(36), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False)
    relevance_score: Mapped[float] = mapped_column(Float, nullable=False)
    reasoning: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    # Relationships
    search_log: Mapped["SearchLog"] = relationship("SearchLog", back_populates="recommendations")
    dataset: Mapped["Dataset"] = relationship("Dataset", back_populates="recommendations")
