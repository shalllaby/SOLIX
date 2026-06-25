import logging
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from backend.tools.dataset_advisor.config import settings
from backend.tools.dataset_advisor.models.database import Base

logger = logging.getLogger("advisor.database")

# Setup SQLAlchemy engine parameters based on database dialect
connect_args = {}
if settings.DATASET_ADVISOR_DB_URL.startswith("sqlite"):
    # SQLite-specific optimization for multi-threaded access
    connect_args = {"check_same_thread": False}

engine = create_async_engine(
    settings.DATASET_ADVISOR_DB_URL,
    connect_args=connect_args,
    echo=False
)

async_session_factory = async_sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
    class_=AsyncSession
)

from sqlalchemy import text

async def init_db() -> None:
    """Initialize database tables."""
    try:
        async with engine.begin() as conn:
            # Enable Foreign Keys in SQLite
            if settings.DATASET_ADVISOR_DB_URL.startswith("sqlite"):
                await conn.execute(text("PRAGMA foreign_keys = ON;"))
            
            # Create all tables if they do not exist
            await conn.run_sync(Base.metadata.create_all)
            
            # Dynamic migration: Add file_size column to datasets table if it doesn't exist yet
            try:
                await conn.execute(text("ALTER TABLE datasets ADD COLUMN file_size VARCHAR(50);"))
                logger.info("Migrated: Added 'file_size' column to 'datasets' table.")
            except Exception:
                # Column already exists or we are initializing from scratch (create_all handled it)
                pass
        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise e

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for obtaining database sessions in FastAPI routes."""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
