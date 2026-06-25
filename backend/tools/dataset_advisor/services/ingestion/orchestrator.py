import logging
from sqlalchemy.ext.asyncio import AsyncSession
from backend.tools.dataset_advisor.services.ingestion.kaggle_provider import KaggleProvider
from backend.tools.dataset_advisor.models.database import Dataset
from backend.tools.dataset_advisor.services.embedding_service import embedding_service
from backend.tools.dataset_advisor.db.qdrant import qdrant_client, QdrantManager
from qdrant_client.http.models import PointStruct

logger = logging.getLogger("advisor.orchestrator")

class IngestOrchestrator:
    async def ingest_all_seeds(self, db: AsyncSession) -> None:
        logger.info("Starting seed dataset ingestion...")
        provider = KaggleProvider()
        seeds = provider._get_fallback_seeds("all")
        
        for seed in seeds:
            # Check if dataset already exists in relational DB
            from sqlalchemy.future import select
            stmt = select(Dataset).where(Dataset.kaggle_id == seed.kaggle_id)
            result = await db.execute(stmt)
            existing = result.scalars().first()
            
            if not existing:
                ds = Dataset(
                    kaggle_id=seed.kaggle_id,
                    title=seed.title,
                    description=seed.description,
                    url=seed.url,
                    row_count=seed.row_count,
                    column_count=seed.column_count,
                    license=seed.license,
                    task_type=seed.task_type,
                    language=seed.language,
                    tags=seed.tags,
                    quality_score=seed.quality_score,
                    file_size=seed.file_size
                )
                db.add(ds)
                await db.flush() # Secure ID
                
                # Upsert into Qdrant vector space
                try:
                    text_to_vectorize = (
                        f"Dataset Title: {ds.title}\n"
                        f"ML Task Type: {ds.task_type}\n"
                        f"Language: {ds.language}\n"
                        f"Tags: {', '.join(ds.tags or [])}\n"
                        f"Description: {ds.description}"
                    )
                    vector = embedding_service.get_embedding(text_to_vectorize)
                    qdrant_client.upsert(
                        collection_name=QdrantManager.COLLECTION_NAME,
                        points=[
                            PointStruct(
                                id=ds.id,
                                vector=vector,
                                payload={
                                    "dataset_id": ds.id,
                                    "task_type": ds.task_type or "tabular",
                                    "language": ds.language or "english",
                                    "row_count": ds.row_count or 1000,
                                    "tags": ds.tags
                                }
                            )
                        ]
                    )
                except Exception as ve:
                    logger.warning(f"Failed Qdrant seeding for '{ds.title}': {ve}")
            
        await db.commit()
        logger.info("Successfully completed seed dataset ingestion.")

ingest_orchestrator = IngestOrchestrator()
