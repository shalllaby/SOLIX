import sys
import os
import asyncio
import time
import logging
from tabulate import tabulate  # We will install this dynamically or use clean fallback string formatting

# Add project root to python path to resolve modules correctly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.app.db.session import init_db, async_session_factory
from backend.app.services.retrieval_pipeline import retrieval_pipeline

# Setup minimal logging to keep console clean
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("advisor.evaluation")

# Define golden evaluation dataset tests
EVAL_TESTS = [
    {
        "name": "Arabic NLP Sentiment Target",
        "query": "أريد داتا صغيرة لتصنيف نصوص المشاعر باللغة العربية",
        "expected_language": "arabic",
        "expected_task": "nlp",
        "max_rows": 15000
    },
    {
        "name": "USA Housing Prices Tabular Regression",
        "query": "I need a dataset for house price prediction in the USA under 10000 rows",
        "expected_language": "english",
        "expected_task": "regression",
        "max_rows": 10000
    },
    {
        "name": "Computer Vision Vehicle Tracking",
        "query": "Computer vision dataset for vehicle detection",
        "expected_language": "english",
        "expected_task": "computer_vision",
        "max_rows": None
    },
    {
        "name": "Arabic Web text large corpus",
        "query": "أريد مجموعة بيانات كبيرة للغة العربية الفصحى",
        "expected_language": "arabic",
        "expected_task": "nlp",
        "max_rows": None
    }
]

async def run_evaluation():
    print("=" * 70)
    print(" RUNNING SEMANTIC RETRIEVAL EVALUATION SUITE")
    print("=" * 70)
    
    try:
        await init_db()
    except Exception as e:
        print(f"[WARNING] Database initialization skipped/locked: {e}")
    
    results = []
    
    for idx, test in enumerate(EVAL_TESTS, 1):
        print(f"\n[{idx}/{len(EVAL_TESTS)}] Evaluating: '{test['name']}'...")
        print(f"   Query: \"{test['query']}\"")
        
        start_time = time.time()
        try:
            async with async_session_factory() as session:
                search_res = await retrieval_pipeline.execute_search(
                    raw_query=test["query"],
                    session_id=f"eval-session-{idx}",
                    db=session
                )
            latency = round((time.time() - start_time) * 1000, 1)
            
            # Check metrics compliance
            datasets = search_res.get("datasets", [])
            hit_rate = 1.0 if len(datasets) > 0 else 0.0
            
            lang_compliance = 0.0
            size_compliance = 0.0
            task_compliance = 0.0
            
            if datasets:
                # Evaluate top result for strict filter matches
                top_ds = datasets[0]
                
                # Language match check
                if top_ds["language"] == test["expected_language"]:
                    lang_compliance = 1.0
                
                # ML Task type match check
                if top_ds["task_type"] == test["expected_task"]:
                    task_compliance = 1.0
                elif test["expected_task"] == "regression" and "price" in top_ds["title"].lower():
                    # Handle title semantics
                    task_compliance = 1.0
                    
                # Row bounds match check
                if test["max_rows"]:
                    rows = top_ds.get("row_count")
                    if rows and rows <= test["max_rows"]:
                        size_compliance = 1.0
                else:
                    size_compliance = 1.0 # No constraint means automatic pass
            
            avg_relevance = round(sum(d["relevance_score"] for d in datasets) / len(datasets), 1) if datasets else 0.0
            
            results.append({
                "Test Target": test["name"],
                "Language": test["expected_language"].upper(),
                "Semantic Hit": "YES" if hit_rate > 0 else "NO",
                "Lang Fit": f"{int(lang_compliance * 100)}%",
                "Task Fit": f"{int(task_compliance * 100)}%",
                "Size Fit": f"{int(size_compliance * 100)}%",
                "Avg Match": f"{avg_relevance}%",
                "Latency (ms)": f"{latency}ms"
            })
            
        except Exception as e:
            print(f"[ERROR] Test crashed: {e}")
            results.append({
                "Test Target": test["name"],
                "Language": test["expected_language"].upper(),
                "Semantic Hit": "CRASH",
                "Lang Fit": "0%",
                "Task Fit": "0%",
                "Size Fit": "0%",
                "Avg Match": "0.0%",
                "Latency (ms)": "0ms"
            })
                
    # Format and print output
    print("\n" + "=" * 70)
    print(" FINAL RETRIEVAL EVALUATION REPORT METRICS")
    print("=" * 70)
    
    # Custom text formatter to avoid external library dependency block on review
    headers = list(results[0].keys())
    rows = [list(r.values()) for r in results]
    
    # Print clean formatted table
    col_widths = [max(len(str(val)) for val in col) for col in zip(*([headers] + rows))]
    
    header_line = " | ".join(f"{h:<{col_widths[i]}}" for i, h in enumerate(headers))
    separator_line = "-+-".join("-" * col_widths[i] for i in range(len(headers)))
    
    print(header_line)
    print(separator_line)
    for row in rows:
        print(" | ".join(f"{str(val):<{col_widths[i]}}" for i, val in enumerate(row)))
    
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(run_evaluation())
