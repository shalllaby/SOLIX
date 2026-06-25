import os
import logging
from typing import List, Optional
from backend.tools.dataset_advisor.config import settings
from backend.tools.dataset_advisor.services.ingestion.base import BaseDatasetProvider, StandardizedDataset

logger = logging.getLogger("advisor.ingestion.kaggle")

class KaggleProvider(BaseDatasetProvider):
    """Kaggle Dataset Provider using official API client with clean fallbacks."""

    @property
    def provider_name(self) -> str:
        return "kaggle"

    def _setup_credentials(self) -> bool:
        """Inject credentials to environment for kaggle-client to authenticate."""
        username = (os.environ.get("KAGGLE_USERNAME") or settings.KAGGLE_USERNAME or "").strip()
        api_key = (os.environ.get("KAGGLE_KEY") or settings.KAGGLE_API_KEY or "").strip()
        
        if not api_key or api_key == "your_api_key_here" or username == "your_username_here":
            logger.warning("Kaggle credentials are not configured or are placeholders.")
            return False
        
        os.environ["KAGGLE_USERNAME"] = username
        os.environ["KAGGLE_KEY"] = api_key
        return True

    async def fetch_datasets(self, query: str, limit: int = 20) -> List[StandardizedDataset]:
        """Fetch datasets from Kaggle API, or fall back to high-fidelity seed data on failure."""
        logger.info(f"KaggleProvider: Fetching datasets for query topic '{query}'")
        
        if not self._setup_credentials():
            logger.warning("Kaggle credentials not established. Falling back to local high-fidelity seeds.")
            return self._get_fallback_seeds(query)

        try:
            from kaggle import KaggleApi
            api = KaggleApi()
            api.authenticate()
            
            logger.info(f"Authenticated Kaggle client. Searching datasets for: {query}")
            datasets_raw = api.dataset_list(search=query, sort_by="hottest")
            
            standard_list = []
            for ds in datasets_raw[:limit]:
                # Safe Tag Parsing
                raw_tags = getattr(ds, 'tags', [])
                tags = []
                if raw_tags:
                    for t in raw_tags:
                        t_name = getattr(t, 'name', None) or (t.get('name') if isinstance(t, dict) else None)
                        if t_name:
                            tags.append(t_name.lower())
                            
                # Determine language constraint based on query or tags
                q_lower = query.lower()
                lang = "english"
                if "arabic" in tags or "arabic" in q_lower or any(ord(c) > 1200 for c in query):
                    lang = "arabic"
                elif any(t in ["multilingual", "translation"] for t in tags) or "multilingual" in q_lower:
                    lang = "multilingual"

                # Guess ML task type
                task = "classification"
                title_lower = ds.title.lower()
                desc_lower = (getattr(ds, 'subtitle', '') or '').lower()
                
                if any(t in ["regression", "prediction", "forecast"] for t in tags) or "price" in title_lower or "house" in title_lower:
                    task = "regression"
                elif any(t in ["nlp", "text", "sentiment", "translation"] for t in tags) or any(x in title_lower or x in desc_lower for x in ["nlp", "text", "sentiment", "summarization", "corpus"]):
                    task = "nlp"
                elif any(t in ["image", "cv", "computer-vision", "detection", "segmentation"] for t in tags) or any(x in title_lower or x in desc_lower for x in ["image", "detection", "vision", "object"]):
                    task = "computer_vision"

                # Parse row/column details from size in bytes
                bytes_val = getattr(ds, 'total_bytes', None) or getattr(ds, 'totalBytes', None)
                estimated_rows = self._estimate_rows(bytes_val, task)
                estimated_cols = getattr(ds, 'columnCount', 10) or 10
                
                # Fetch size string or estimate it
                raw_size = getattr(ds, 'size', None)
                file_size_str = None
                if raw_size:
                    size_str = str(raw_size).strip()
                    import re
                    match = re.match(r"(\d+(?:\.\d+)?)\s*([a-zA-Z]+)", size_str)
                    if match:
                        file_size_str = f"{match.group(1)} {match.group(2).upper()}"
                    else:
                        file_size_str = size_str
                else:
                    file_size_str = self._format_size(bytes_val)
                    
                if not file_size_str or file_size_str == "Unknown size":
                    file_size_str = self._fallback_size_from_rows(estimated_rows, task)

                # Fetch license and usability safely
                license_name = getattr(ds, 'license_name', None) or getattr(ds, 'licenseName', 'Unknown')
                usability = getattr(ds, 'usability_rating', None) or getattr(ds, 'usabilityRating', 0.6)
                quality_score = float(usability) * 10.0 if usability else 6.0
                
                # Extract descriptive description/subtitle
                description = getattr(ds, 'subtitle', '') or f"High-quality dataset for {ds.title} on Kaggle."
                if len(description) < 10:
                    description = f"Real-world Kaggle dataset for {ds.title}. Ref: {ds.ref}"

                # Build unified schema record
                standard_list.append(StandardizedDataset(
                    title=ds.title,
                    description=description,
                    url=f"https://www.kaggle.com/datasets/{ds.ref}",
                    kaggle_id=ds.ref,
                    provider=self.provider_name,
                    row_count=estimated_rows,
                    column_count=estimated_cols,
                    license=license_name,
                    task_type=task,
                    language=lang,
                    tags=tags if tags else [query, lang, task],
                    quality_score=quality_score,
                    file_size=file_size_str
                ))
            
            if not standard_list:
                logger.warning(f"Kaggle API returned empty list for '{query}'. Loading mock fallback seed datasets.")
                return self._get_fallback_seeds(query)

            logger.info(f"Ingested {len(standard_list)} datasets successfully from Kaggle API.")
            return standard_list

        except Exception as e:
            logger.error(f"Kaggle API ingestion failed: {e}. Falling back to high-fidelity seed datasets.")
            return self._get_fallback_seeds(query)

    def _format_size(self, bytes_val: Optional[int]) -> str:
        """Format size in bytes to a human-readable string."""
        if not bytes_val:
            return "Unknown size"
        try:
            val = float(bytes_val)
            for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
                if val < 1024.0:
                    if val.is_integer() or abs(val - round(val)) < 0.05:
                        return f"{int(round(val))} {unit}"
                    return f"{val:.1f} {unit}"
                val /= 1024.0
            return f"{val:.1f} PB"
        except Exception:
            return "Unknown size"

    def _fallback_size_from_rows(self, rows: int, task: str) -> str:
        """Estimate file size from estimated row counts if actual bytes/size is missing."""
        if task == "nlp":
            bytes_val = rows * 400
        elif task == "computer_vision":
            bytes_val = rows * 150 * 1024
        else:
            bytes_val = rows * 120
        return self._format_size(bytes_val)

    def _estimate_rows(self, bytes_val: Optional[int], task: str) -> int:
        """Estimate row count based on file size in bytes and task profile."""
        if not bytes_val:
            return 5000
        
        try:
            # Heuristic: average text row is 400 bytes, tabular row is 120 bytes, CV is 150KB per image
            if task == "nlp":
                return max(100, int(bytes_val / 400))
            elif task == "computer_vision":
                return max(50, int(bytes_val / (150 * 1024)))
            else:  # Tabular / Regression
                return max(200, int(bytes_val / 120))
        except:
            return 8000

    def _get_fallback_seeds(self, query: str) -> List[StandardizedDataset]:
        """Provides a curated set of bilingual/multilingual datasets if Kaggle is offline or keys are empty."""
        logger.info("Serving localized seed datasets.")
        seeds = [
            StandardizedDataset(
                title="Arabic Sentiment Analysis Dataset (ASTD)",
                description="A high-quality dataset containing Egyptian tweets classified for sentiment analysis. Perfect for training NLP models, parsing emotions, and learning Arabic text preprocessing.",
                url="https://www.kaggle.com/datasets/maksut/arabic-sentiment-analysis",
                kaggle_id="maksut/arabic-sentiment-analysis",
                provider=self.provider_name,
                row_count=10000,
                column_count=3,
                license="CC BY-NC-SA 4.0",
                task_type="nlp",
                language="arabic",
                tags=["arabic", "sentiment", "nlp", "classification", "tweets"],
                quality_score=8.5,
                file_size="1.4 MB"
            ),
            StandardizedDataset(
                title="USA House Price Prediction Dataset",
                description="Contains historical real estate transactional data across the United States. Key attributes include bedrooms, bathrooms, square footage, zip code, and selling price. Ideal for linear regression and tabular ML algorithms.",
                url="https://www.kaggle.com/datasets/paultimothymooney/usa-housing-dataset",
                kaggle_id="paultimothymooney/usa-housing-dataset",
                provider=self.provider_name,
                row_count=5000,
                column_count=7,
                license="Open Database License (ODbL)",
                task_type="regression",
                language="english",
                tags=["regression", "tabular", "housing", "price-prediction", "tabular-ml"],
                quality_score=9.2,
                file_size="320 KB"
            ),
            StandardizedDataset(
                title="Arabic Text Summarization Corpus",
                description="A rich corpus of standard Arabic news articles paired with expert manual summaries. Designed for sequence-to-sequence model training, abstractive summarization, and NLP benchmarking.",
                url="https://www.kaggle.com/datasets/gowhar/arabic-news-summarization",
                kaggle_id="gowhar/arabic-news-summarization",
                provider=self.provider_name,
                row_count=1200,
                column_count=2,
                license="MIT License",
                task_type="nlp",
                language="arabic",
                tags=["arabic", "nlp", "text-summarization", "news", "generative"],
                quality_score=7.8,
                file_size="650 KB"
            ),
            StandardizedDataset(
                title="Vehicle Detection and Tracking Dataset",
                description="Computer vision dataset consisting of labeled images of road traffic. Includes bounding boxes for cars, buses, trucks, and motorcycles under various weather conditions. Supports YOLO v8/v9 formats.",
                url="https://www.kaggle.com/datasets/brsdincer/vehicle-detection-image-dataset",
                kaggle_id="brsdincer/vehicle-detection-image-dataset",
                provider=self.provider_name,
                row_count=4500,
                column_count=4,
                license="CC0: Public Domain",
                task_type="computer_vision",
                language="english",
                tags=["cv", "image-classification", "computer-vision", "yolo", "vehicles"],
                quality_score=9.0,
                file_size="124.5 MB"
            ),
            StandardizedDataset(
                title="Global Customer Churn Dataset",
                description="Tabular classification dataset tracking telecommunication consumer behaviors. Includes tenure, monthly charges, contract types, and a binary churn label. Used for predicting customer attrition.",
                url="https://www.kaggle.com/datasets/shantanudg/telecom-customer-churn-dataset",
                kaggle_id="shantanudg/telecom-customer-churn-dataset",
                provider=self.provider_name,
                row_count=7032,
                column_count=21,
                license="Apache 2.0",
                task_type="classification",
                language="english",
                tags=["classification", "tabular", "customer-churn", "behavior", "tabular-ml"],
                quality_score=8.7,
                file_size="480 KB"
            ),
            StandardizedDataset(
                title="Arabic Language Web Text Corpus",
                description="A massive crawl of Arabic web forums, news portals, and social blogs. Highly suited for pre-training large language models (LLMs), training word embeddings, and linguistic statistics research.",
                url="https://www.kaggle.com/datasets/almarri/arabic-web-corpus",
                kaggle_id="almarri/arabic-web-corpus",
                provider=self.provider_name,
                row_count=95000,
                column_count=2,
                license="CC BY-ND 4.0",
                task_type="nlp",
                language="arabic",
                tags=["arabic", "corpus", "nlp", "web-crawl", "unsupervised"],
                quality_score=8.1,
                file_size="14.8 MB"
            ),
            StandardizedDataset(
                title="Heart Disease Classification Database",
                description="Patient clinical metrics database detailing heart rate, chest pain types, cholesterol, blood pressure, and binary heart disease labels. Recommended for beginners practicing classification models.",
                url="https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset",
                kaggle_id="johnsmith88/heart-disease-dataset",
                provider=self.provider_name,
                row_count=1025,
                column_count=14,
                license="Public Domain",
                task_type="classification",
                language="english",
                tags=["classification", "tabular", "medical", "heart-disease", "health"],
                quality_score=9.5,
                file_size="24 KB"
            )
        ]
        
        q = query.lower()
        filtered = [
            s for s in seeds 
            if q in s.title.lower() 
            or q in s.description.lower() 
            or any(q in tag for tag in s.tags)
            or (("arabic" in q or "عرب" in q) and s.language == "arabic")
            or (("nlp" in q or "text" in q or "نصوص" in q) and s.task_type == "nlp")
            or (("cv" in q or "image" in q or "صورة" in q) and s.task_type == "computer_vision")
            or (("price" in q or "regression" in q or "سعر" in q) and s.task_type == "regression")
        ]
        
        return filtered if filtered else seeds
