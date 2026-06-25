import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import func

from backend.app.db.session import get_db
from backend.app.models.database import Dataset, SearchLog, Recommendation
from backend.app.models.schemas import SearchRequest, SearchResponse, StatsResponse
from backend.app.services.retrieval_pipeline import retrieval_pipeline
from backend.app.db.qdrant import qdrant_client

logger = logging.getLogger("advisor.routes")
router = APIRouter(prefix="/api")

@router.post("/search", response_model=SearchResponse)
async def search_datasets(
    request: SearchRequest,
    db: AsyncSession = Depends(get_db)
):
    """Conversational search endpoint parsing intents, matching semantic paths, and explaining reasons."""
    logger.info(f"Route: POST /search - query: '{request.query}'")
    try:
        results = await retrieval_pipeline.execute_search(
            raw_query=request.query,
            session_id=request.session_id,
            db=db
        )
        return results
    except Exception as e:
        logger.error(f"Search endpoint execution failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An internal error occurred during retrieval: {str(e)}"
        )

@router.get("/stats", response_model=StatsResponse)
async def get_system_stats(
    db: AsyncSession = Depends(get_db)
):
    """Retrieve internal metadata and database statistics."""
    try:
        # SQL count fetches
        ds_count = (await db.execute(select(func.count()).select_from(Dataset))).scalar_one()
        log_count = (await db.execute(select(func.count()).select_from(SearchLog))).scalar_one()
        rec_count = (await db.execute(select(func.count()).select_from(Recommendation))).scalar_one()
        
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
