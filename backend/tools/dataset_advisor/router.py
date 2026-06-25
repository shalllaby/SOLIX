import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func

from backend.tools.dataset_advisor.db.session import get_db
from backend.tools.dataset_advisor.models.database import Dataset, SearchLog, Recommendation
from backend.tools.dataset_advisor.models.schemas import SearchRequest, SearchResponse, StatsResponse
from backend.tools.dataset_advisor.services.retrieval_pipeline import retrieval_pipeline
from backend.tools.dataset_advisor.db.qdrant import qdrant_client

from backend.auth import get_current_user
from backend.models import User

logger = logging.getLogger("advisor.routes")
dataset_advisor_router = APIRouter(prefix="/api/dataset-advisor", tags=["Dataset Advisor"])

from backend.middleware.barrier import CredentialsBarrier

@dataset_advisor_router.post("/search", response_model=SearchResponse)
async def search_datasets(
    request: SearchRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
    _barrier = Depends(CredentialsBarrier(["groq_api_key"]))
):
    """Conversational search endpoint parsing intents, matching semantic paths, and explaining reasons."""
    logger.info(f"Route: POST /api/dataset-advisor/search - query: '{request.query}' - user: {current_user.id}")
    try:
        from backend.database import SessionLocal
        from backend.models import UserSettings
        db_main = SessionLocal()
        settings = db_main.query(UserSettings).filter_by(user_id=current_user.id).first()
        groq_api_key = settings.groq_api_key if settings else None
        db_main.close()

        results = await retrieval_pipeline.execute_search(
            raw_query=request.query,
            session_id=request.session_id,
            db=db,
            user_id=current_user.id,
            api_key=groq_api_key
        )
        return results
    except Exception as e:
        logger.error(f"Search endpoint execution failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An internal error occurred during retrieval: {str(e)}"
        )

@dataset_advisor_router.get("/stats", response_model=StatsResponse)
async def get_system_stats(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Retrieve internal metadata and database statistics."""
    try:
        # SQL count fetches
        ds_count = (await db.execute(select(func.count()).select_from(Dataset))).scalar_one()
        log_count = (await db.execute(select(func.count()).select_from(SearchLog).filter(SearchLog.user_id == current_user.id))).scalar_one()
        rec_count = (await db.execute(
            select(func.count())
            .select_from(Recommendation)
            .join(SearchLog, Recommendation.log_id == SearchLog.id)
            .filter(SearchLog.user_id == current_user.id)
        )).scalar_one()
        
        # Test Qdrant connectivity
        try:
            q_collections = qdrant_client.get_collections().collections
            q_status = f"Connected ({len(q_collections)} collections)"
        except Exception as e:
            q_status = f"Offline/Error: {str(e)}"
            
        # SQL engine dialect check
        dialect = db.bind.dialect.name
        
        return StatsResponse(
            datasets_count=ds_count,
            search_logs_count=log_count,
            recommendations_count=rec_count,
            qdrant_status=q_status,
            engine_dialect=dialect
        )
    except Exception as e:
        logger.error(f"Failed to fetch system stats: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Database monitoring error: {str(e)}"
        )

