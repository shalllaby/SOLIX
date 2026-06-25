import logging
from typing import Optional, List
from pydantic import BaseModel, Field
from backend.app.services.llm_service import llm_service

logger = logging.getLogger("advisor.intent_parser")

class ParsedQueryIntent(BaseModel):
    """Pydantic model representing structured filters extracted from the user's raw query."""
    task_type: Optional[str] = Field(
        None, 
        description="Machine learning task type: e.g. classification, regression, clustering, generation, sentiment-analysis, object-detection, prediction, etc."
    )
    modality: Optional[str] = Field(
        None, 
        description="Data format modality: e.g. nlp, vision, tabular, audio, time-series, graph, multimodal"
    )
    language: Optional[str] = Field(
        None, 
        description="Target dataset language constraint (strictly 'arabic', 'english', or 'multilingual'). If not specified, leave null."
    )
    max_rows: Optional[int] = Field(
        None, 
        description="Maximum rows requested if explicitly mentioned (e.g. 'under 10000 rows', 'small dataset' -> 10000). Otherwise null."
    )
    min_rows: Optional[int] = Field(
        None, 
        description="Minimum rows requested if explicitly mentioned (e.g. 'at least 5000 rows', 'large dataset' -> 5000). Otherwise null."
    )
    topic: Optional[str] = Field(
        None, 
        description="Clean core dataset topic extracted from query (e.g. 'house price', 'diabetes', 'vehicle detection', 'sentiment')."
    )
    constraints: Optional[str] = Field(
        None, 
        description="Other textual or numeric constraints (e.g., 'Egyptian', 'under 5000 rows')."
    )
    primary_kaggle_query: str = Field(
        ...,
        description="A simple, direct English keyword search query optimized for Kaggle API. Keep it short and specific (e.g. 'house price', 'diabetes', 'vehicle detection'). Avoid long descriptive text."
    )
    expansion_queries: List[str] = Field(
        default_factory=list,
        description="Lightweight English keyword synonyms or query expansions (1-3 simple queries, e.g. ['housing price', 'real estate prices']) to improve coverage."
    )
    semantic_search_query: str = Field(
        "",
        description="Reconstructed and translation-aligned semantic query in English optimized for backward compatibility. Set equal to primary_kaggle_query."
    )
    detected_language: str = Field(
        "english",
        description="Language of the user's input query. Strictly 'english', 'arabic', or 'multilingual'."
    )


class QueryUnderstandingPipeline:
    """Service to parse raw natural language queries into machine-readable search filters."""
    
    SYSTEM_INSTRUCTION = (
        "You are an expert ML dataset retrieval agent. Your goal is to dissect a user's dataset search query, "
        "detect their underlying machine learning intent, translate semantic terms to English, "
        "and extract numeric/categorical filters. You support both English and Arabic. "
        "Produce structural metadata matching the schema. "
        "Generate short, direct, keyword-based English queries for Kaggle search (e.g. 'house price', 'vehicle detection'). "
        "Do NOT write long descriptions as the search query. "
        "For semantic_search_query, set it to the same value as primary_kaggle_query."
    )

    async def parse_query(self, raw_query: str) -> ParsedQueryIntent:
        """Parse raw query asynchronously."""
        logger.info(f"Parsing raw user query: '{raw_query}'")
        
        prompt = (
            f"Understand and parse the following dataset query:\n"
            f"Query: \"{raw_query}\"\n\n"
            f"Steps:\n"
            f"1. Detect the intent, target tasks, modality, and language requirements.\n"
            f"2. Extract dataset size bounds if explicitly mentioned.\n"
            f"3. Identify the clean topic (e.g., 'house price' or 'sentiment') and any other constraints.\n"
            f"4. Generate a 'primary_kaggle_query' that is a short, direct, 2-4 word English search term.\n"
            f"5. Generate 1-3 simple English 'expansion_queries' (synonyms/variations) to broaden coverage.\n"
            f"6. Populate 'semantic_search_query' with the value of 'primary_kaggle_query'."
        )

        try:
            parsed_intent = await llm_service.chat_completion_json(
                prompt=prompt,
                response_model=ParsedQueryIntent,
                system_instruction=self.SYSTEM_INSTRUCTION,
                temperature=0.1
            )
            # Ensure semantic_search_query is set
            if not parsed_intent.semantic_search_query:
                parsed_intent.semantic_search_query = parsed_intent.primary_kaggle_query
            
            logger.info(f"Successfully extracted filters: {parsed_intent.model_dump()}")
            return parsed_intent
        except Exception as e:
            logger.error(f"Error parsing intent: {e}. Falling back to default search intent.")
            # Graceful degradation fallback
            is_arabic = any(ord(char) > 1200 for char in raw_query)
            detected_lang = "arabic" if is_arabic else "english"
            
            # Simple fallback translation rules
            fallback_query = "arabic sentiment analysis" if is_arabic and "شاعر" in raw_query else raw_query
            if "بيت" in raw_query or "منزل" in raw_query:
                fallback_query = "house price"
            elif "سيار" in raw_query or "مركب" in raw_query:
                fallback_query = "vehicle detection"
                
            return ParsedQueryIntent(
                primary_kaggle_query=fallback_query,
                semantic_search_query=fallback_query,
                detected_language=detected_lang,
                expansion_queries=[fallback_query + " dataset"]
            )

# Global instantiator
query_parser = QueryUnderstandingPipeline()
