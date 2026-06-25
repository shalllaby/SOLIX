import logging
from typing import List, Dict, Any, Optional
import numpy as np
from pydantic import BaseModel
from sqlalchemy.future import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.config import settings
from backend.app.models.database import Dataset, SearchLog, Recommendation
from backend.app.services.query_understanding import ParsedQueryIntent, query_parser
from backend.app.services.embedding_service import embedding_service
from backend.app.services.llm_service import llm_service
from backend.app.db.qdrant import qdrant_client, QdrantManager
from backend.app.services.ingestion.kaggle_provider import KaggleProvider
from backend.app.services.cache_service import semantic_cache

logger = logging.getLogger("advisor.retrieval")

class DatasetRecommendationResult(BaseModel):
    """Pydantic schema representing the complete payload for a recommended dataset card."""
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


class RetrievalPipeline:
    """Core semantic intelligence engine coordinating search, live Kaggle query, filtering, scoring, and reasoning."""

    async def execute_search(
        self, 
        raw_query: str, 
        session_id: str, 
        db: AsyncSession
    ) -> Dict[str, Any]:
        """Execute the entire async live retrieval flow."""
        logger.info(f"RetrievalPipeline: Initiating live search for session '{session_id}' -> '{raw_query}'")
        
        # 0. Check Semantic Cache
        cached_res = semantic_cache.get(raw_query)
        if cached_res:
            logger.info(f"RetrievalPipeline: Semantic Cache HIT for '{raw_query}'. Returning cached payload.")
            return cached_res
            
        # 1. Understanding & intent extraction
        intent: ParsedQueryIntent = await query_parser.parse_query(raw_query)
        
        # Log session query properties
        search_log = SearchLog(
            session_id=session_id,
            query_text=raw_query,
            detected_language=intent.detected_language,
            extracted_filters=intent.model_dump()
        )
        db.add(search_log)
        await db.flush() # Secure log_id
        
        # 2. Live Kaggle Retrieval (Primary Source)
        provider = KaggleProvider()
        candidates = []
        
        # Fetch using primary simple Kaggle query
        try:
            logger.info(f"RetrievalPipeline: Fetching from Kaggle using primary query '{intent.primary_kaggle_query}'")
            primary_results = await provider.fetch_datasets(intent.primary_kaggle_query, limit=15)
            candidates.extend(primary_results)
        except Exception as e:
            logger.error(f"Failed live Kaggle search for primary query '{intent.primary_kaggle_query}': {e}")

        # Fetch using expansion queries if coverage is small or query is specific
        for exp_query in intent.expansion_queries:
            if len(candidates) >= 15:
                break
            if exp_query and exp_query.strip().lower() != intent.primary_kaggle_query.strip().lower():
                try:
                    logger.info(f"RetrievalPipeline: Fetching from Kaggle using expansion query '{exp_query}'")
                    expansion_results = await provider.fetch_datasets(exp_query, limit=10)
                    candidates.extend(expansion_results)
                except Exception as e:
                    logger.error(f"Failed live Kaggle search for expansion query '{exp_query}': {e}")

        # Deduplicate retrieved candidates by kaggle_id or url
        seen_keys = set()
        unique_candidates = []
        for c in candidates:
            key = c.kaggle_id or c.url
            if key not in seen_keys:
                seen_keys.add(key)
                unique_candidates.append(c)

        # 3. Intelligent Fallback: Seed/Local Database matches if live search yields nothing
        if not unique_candidates:
            logger.warning("Live Kaggle search returned 0 candidates. Falling back to high-fidelity seed datasets.")
            try:
                seed_matches = provider._get_fallback_seeds(raw_query)
                unique_candidates.extend(seed_matches)
            except Exception as e:
                logger.error(f"Error fetching fallback seeds: {e}")

        # 4. Save/Update records in SQL DB dynamically to get local IDs
        db_datasets = []
        for candidate in unique_candidates:
            try:
                stmt = select(Dataset).where(
                    (Dataset.kaggle_id == candidate.kaggle_id) if candidate.kaggle_id else (Dataset.url == candidate.url)
                )
                result = await db.execute(stmt)
                existing_ds = result.scalars().first()

                if existing_ds:
                    # Update metadata on the fly if needed
                    existing_ds.row_count = candidate.row_count or existing_ds.row_count
                    existing_ds.column_count = candidate.column_count or existing_ds.column_count
                    existing_ds.quality_score = max(candidate.quality_score, existing_ds.quality_score)
                    if candidate.language:
                        existing_ds.language = candidate.language
                    if candidate.task_type:
                        existing_ds.task_type = candidate.task_type
                    db_ds = existing_ds
                else:
                    db_ds = Dataset(
                        kaggle_id=candidate.kaggle_id,
                        title=candidate.title,
                        description=candidate.description,
                        url=candidate.url,
                        row_count=candidate.row_count,
                        column_count=candidate.column_count,
                        license=candidate.license,
                        task_type=candidate.task_type,
                        language=candidate.language,
                        tags=candidate.tags,
                        quality_score=candidate.quality_score
                    )
                    db.add(db_ds)
                    await db.flush() # Forces local ID generation
                db_datasets.append(db_ds)
            except Exception as e:
                logger.error(f"Failed to cache dataset '{candidate.title}' in SQL: {e}")

        # 5. Optional Qdrant caching/acceleration (failsafe)
        try:
            for ds in db_datasets:
                text_to_vectorize = (
                    f"Dataset Title: {ds.title}\n"
                    f"ML Task Type: {ds.task_type}\n"
                    f"Language: {ds.language}\n"
                    f"Tags: {', '.join(ds.tags or [])}\n"
                    f"Description: {ds.description}"
                )
                vector = embedding_service.get_embedding(text_to_vectorize)
                
                from qdrant_client.http.models import PointStruct
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
            logger.info(f"Succeeded in writing optional semantic caching for {len(db_datasets)} datasets in Qdrant.")
        except Exception as e:
            logger.warning(f"Optional Qdrant caching / acceleration skipped: {e}")

        # 6. Advanced AI Post-Processing & Reranking Layer
        # Rerank is a smart hybrid function: Semantic embedding alignment + task matching + modality match + row constraint mapping
        query_vector = embedding_service.get_embedding(intent.primary_kaggle_query)
        ranked_candidates = []

        for ds in db_datasets:
            # A. Compute semantic score between primary query and dataset description details
            try:
                ds_text = f"{ds.title} {ds.description or ''} {' '.join(ds.tags or [])}"
                ds_vector = embedding_service.get_embedding(ds_text)
                
                vec_q = np.array(query_vector)
                vec_d = np.array(ds_vector)
                dot_product = np.dot(vec_q, vec_d)
                norm_q = np.linalg.norm(vec_q)
                norm_d = np.linalg.norm(vec_d)
                
                semantic_score = float(dot_product / (norm_q * norm_d)) if norm_q > 0 and norm_d > 0 else 0.5
                semantic_score = max(0.0, min(1.0, semantic_score))
            except Exception as e:
                logger.error(f"Semantic scoring calculation failed: {e}")
                semantic_score = 0.5

            # B. Task matching score
            task_fit = 0.0
            if intent.task_type and ds.task_type:
                it_lower = intent.task_type.lower()
                dt_lower = ds.task_type.lower()
                if it_lower == dt_lower:
                    task_fit = 1.0
                elif it_lower in dt_lower or dt_lower in it_lower:
                    task_fit = 0.7
                elif (it_lower == "regression" and "price" in ds.title.lower()) or (it_lower == "nlp" and "text" in ds.title.lower()):
                    task_fit = 0.8
            else:
                task_fit = 0.7 # neutral broad match

            # C. Modality matching score (detect fit based on metadata/tags/descriptions)
            modality_fit = 0.0
            if intent.modality:
                im_lower = intent.modality.lower()
                ds_task = (ds.task_type or "").lower()
                tags_str = " ".join(ds.tags or []).lower()
                desc_str = (ds.description or "").lower()
                
                if im_lower == "tabular" and (any(x in tags_str or x in desc_str for x in ["tabular", "csv", "excel", "regression", "dataframe", "columns"]) or ds_task in ["regression", "classification"]):
                    modality_fit = 1.0
                elif im_lower == "nlp" and (any(x in tags_str or x in desc_str for x in ["nlp", "text", "sentiment", "summarization", "corpus", "translation", "language"]) or ds_task == "nlp"):
                    modality_fit = 1.0
                elif im_lower == "vision" and (any(x in tags_str or x in desc_str for x in ["image", "cv", "computer-vision", "detection", "yolo", "segmentation", "images"]) or ds_task == "computer_vision"):
                    modality_fit = 1.0
                else:
                    modality_fit = 0.3 # low match
            else:
                modality_fit = 0.8 # broad fit

            # D. Language matching score (multilingual matching logic)
            lang_fit = 1.0
            if intent.language:
                il_lower = intent.language.lower()
                dl_lower = (ds.language or "").lower()
                if il_lower == dl_lower:
                    lang_fit = 1.0
                elif il_lower == "arabic" and ("arabic" in dl_lower or "arabic" in " ".join(ds.tags or []).lower() or any(ord(c) > 1200 for c in (ds.description or ""))):
                    lang_fit = 1.0
                elif il_lower == "arabic":
                    lang_fit = 0.1 # Arabic highly penalized if not matching
                elif il_lower == "english" and "english" not in dl_lower:
                    lang_fit = 0.6
            else:
                # If query contains Arabic letters, boost Arabic datasets
                query_is_arabic = any(ord(char) > 1200 for char in raw_query)
                ds_is_arabic = "arabic" in (ds.language or "").lower() or any(ord(c) > 1200 for c in (ds.description or ""))
                if query_is_arabic and ds_is_arabic:
                    lang_fit = 1.0
                elif query_is_arabic:
                    lang_fit = 0.4
                    
            # E. Row constraints compliance score (Continuous Progressive Penalty)
            size_fit = 1.0
            if intent.max_rows and ds.row_count:
                if ds.row_count <= intent.max_rows:
                    size_fit = 1.0
                else:
                    # progressive penalty ratio
                    ratio = ds.row_count / intent.max_rows
                    if ratio <= 1.5:
                        size_fit = 0.8
                    elif ratio <= 3.0:
                        size_fit = 0.5
                    elif ratio <= 10.0:
                        size_fit = 0.2
                    else:
                        size_fit = 0.05
            if intent.min_rows and ds.row_count:
                if ds.row_count >= intent.min_rows:
                    size_fit = 1.0
                else:
                    size_fit = 0.4

            # Combined weighted score:
            # 40% semantic content, 20% task fit, 15% modality, 15% language, 10% row bounds size compliance
            final_weighted_score = (
                0.40 * semantic_score +
                0.20 * task_fit +
                0.15 * modality_fit +
                0.15 * lang_fit +
                0.10 * size_fit
            )
            
            relevance_percentage = round(final_weighted_score * 100, 1)
            ranked_candidates.append((ds, relevance_percentage))

        # Sort descending by custom calculated score
        ranked_candidates.sort(key=lambda x: x[1], reverse=True)
        
        # Select top 3 datasets for full reasoning generation
        top_candidates = ranked_candidates[:3]

        # 7. Asynchronously generate context-aware AI reasoning in Arabic or English
        recommendations = await self._generate_reasoning_batch(
            top_candidates, 
            raw_query, 
            intent.detected_language
        )
        
        # 8. Log recommendations to Relational DB
        for rec in recommendations:
            try:
                db_rec = Recommendation(
                    log_id=search_log.id,
                    dataset_id=rec.id,
                    relevance_score=rec.relevance_score,
                    reasoning=rec.reasoning
                )
                db.add(db_rec)
            except Exception as e:
                logger.error(f"Error logging recommendation record: {e}")

        # 9. Generate friendly conversational intro reply
        advisor_message = await self._generate_conversational_reply(
            recommendations, 
            raw_query, 
            intent.detected_language
        )
        try:
            await db.commit()
        except Exception as db_err:
            logger.warning(f"Database transaction commit skipped or locked: {db_err}")
            try:
                await db.rollback()
            except Exception:
                pass
        
        # Formulate payload and store in semantic cache before return
        result_payload = {
            "query_analysis": intent.model_dump(),
            "detected_language": intent.detected_language,
            "advisor_message": advisor_message,
            "datasets": [rec.model_dump() for rec in recommendations]
        }
        semantic_cache.set(raw_query, result_payload)
        
        return result_payload

    async def _generate_reasoning_batch(
        self, 
        candidates: List[tuple], 
        raw_query: str, 
        language: str
    ) -> List[DatasetRecommendationResult]:
        """Query LLM to generate custom context-aware reasoning details for top matched items."""
        results = []
        for ds, score in candidates:
            # Language localization instructions
            if language == "arabic":
                system_instruction = (
                    "أنت خبير وباحث بيانات ذكي. اشرح للمستخدم باختصار شديد جداً (لا يتعدى سطرين) "
                    "لماذا هذه البيانات مناسبة لهدف بحثه."
                )
                prompt = (
                    f"الطلب: \"{raw_query}\"\n"
                    f"اسم مجموعة البيانات: \"{ds.title}\"\n"
                    f"الوصف: \"{ds.description}\"\n"
                    f"النوع: \"{ds.task_type}\"\n"
                    f"عدد الصفوف: {ds.row_count or 'غير معروف'}\n\n"
                    f"اكتب التبرير باللغة العربية بأسلوب احترافي مقنع وموجز للغاية."
                )
            else:
                system_instruction = (
                    "You are a professional AI dataset consultant. Explain to the user in 1-2 concise, "
                    "convincing sentences exactly why this dataset fits their specific project goal."
                )
                prompt = (
                    f"User Intent: \"{raw_query}\"\n"
                    f"Dataset Title: \"{ds.title}\"\n"
                    f"Description: \"{ds.description}\"\n"
                    f"Task: \"{ds.task_type}\"\n"
                    f"Row count: {ds.row_count or 'Unknown'}\n\n"
                    f"Provide a brief, compelling, and context-aware reason."
                )

            try:
                reasoning = await llm_service.chat_completion(
                    prompt=prompt,
                    system_instruction=system_instruction,
                    temperature=0.3
                )
            except Exception as e:
                logger.error(f"Reasoning LLM call failed: {e}")
                reasoning = (
                    "هذه البيانات مطابقة جداً لمعايير بحثك من حيث المهمة واللغة وحجم البيانات."
                    if language == "arabic" else
                    "This dataset matches your search criteria based on modality, language, and row dimensions."
                )
            
            results.append(DatasetRecommendationResult(
                id=ds.id,
                title=ds.title,
                description=ds.description,
                url=ds.url,
                row_count=ds.row_count,
                column_count=ds.column_count,
                license=ds.license,
                task_type=ds.task_type,
                language=ds.language,
                tags=ds.tags or [],
                relevance_score=score,
                reasoning=reasoning.strip()
            ))
            
        return results

    async def _generate_conversational_reply(
        self, 
        recommendations: List[DatasetRecommendationResult], 
        raw_query: str, 
        language: str
    ) -> str:
        """Generate a friendly, concise summarization advisor response outlining the matches."""
        if not recommendations:
            return (
                "عذراً، لم أجد أي مجموعات بيانات تطابق معايير بحثك حالياً. هل ترغب في تعديل فلاتر البحث أو الكلمات المفتاحية؟"
                if language == "arabic" else
                "Sorry, I couldn't find any datasets matching your search criteria. Try modifying your filters or terms."
            )
            
        titles = ", ".join([f"'{rec.title}'" for rec in recommendations])
        
        if language == "arabic":
            system_instruction = (
                "أنت مستشار داتا ذكي ومرحب. لخص ردك بأسلوب ودود ومهني وبلسنتك سياق التوصيات. "
                "رحب بالمستخدم باللغة العربية واشرح له باختصار شديد جداً أنك قمت باختيار وتصفية أفضل مجموعات البيانات لمشروعه."
            )
            prompt = (
                f"طلب البحث: \"{raw_query}\"\n"
                f"أفضل التوصيات: {titles}\n\n"
                f"اكتب فقرة واحدة قصيرة وملخصة باللغة العربية للترحيب وتقديم هذه البيانات."
            )
        else:
            system_instruction = (
                "You are a friendly, highly intelligent conversational dataset recommender. "
                "Acknowledge the user's specific ML project goal and introduce the datasets you've found."
            )
            prompt = (
                f"Search Query: \"{raw_query}\"\n"
                f"Selected recommendations: {titles}\n\n"
                f"Write a friendly 1-2 sentence response introducing these matches."
            )

        try:
            return await llm_service.chat_completion(
                prompt=prompt,
                system_instruction=system_instruction,
                temperature=0.4
            )
        except Exception:
            return (
                f"لقد قمت بتحليل طلبك ووجدت {len(recommendations)} مجموعات بيانات ممتازة تناسب احتياجاتك تماماً:"
                if language == "arabic" else
                f"I parsed your query and found {len(recommendations)} high-quality datasets that fit your requirements perfectly:"
            )

# Global instantiator
retrieval_pipeline = RetrievalPipeline()
