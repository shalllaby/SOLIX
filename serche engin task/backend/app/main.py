import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.future import select
from sqlalchemy import func

from backend.app.config import settings
from backend.app.db.session import init_db, async_session_factory
from backend.app.models.database import Dataset
from backend.app.api.routes import router
from backend.app.services.ingestion.orchestrator import ingest_orchestrator

# Setup clean global logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("advisor.main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle events handling automatic database initialization and seeding."""
    logger.info("Initializing AI Dataset Advisor backend application lifespan...")
    
    # 1. Initialize Relational & Vector tables
    await init_db()
    
    # 2. Check if DB has seeds, if empty automatically seed to ensure instant out-of-the-box runtime
    async with async_session_factory() as session:
        try:
            stmt = select(func.count()).select_from(Dataset)
            result = await session.execute(stmt)
            count = result.scalar_one()
            
            if count == 0:
                logger.info("Database contains 0 datasets. Auto-triggering seed ingestion pipeline...")
                await ingest_orchestrator.ingest_all_seeds(session)
            else:
                logger.info(f"Database initialized with {count} pre-existing datasets. Skipping auto-seeding.")
        except Exception as e:
            logger.error(f"Failed during lifespan database check/seeding: {e}")
            
    yield
    logger.info("Shutting down backend lifespan...")

# Create FastAPI Instance
app = FastAPI(
    title=settings.APP_NAME,
    description="An AI-native conversational dataset search and recommendation platform.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS policies for monorepo frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this to specific domain URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Routers
app.include_router(router)

@app.get("/")
def read_root():
    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "message": "AI Dataset Advisor API is running successfully."
    }
