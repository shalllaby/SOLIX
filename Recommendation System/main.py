from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import io

# Import the MLAdvisor recommendations function
from ml_advisor import get_recommendations

app = FastAPI(
    title="MLAdvisor API",
    description="API for providing Machine Learning model recommendations based on dataset metadata.",
    version="1.0.0"
)

# Enable CORS for all domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "message": "MLAdvisor API is running properly."}

@app.post("/recommend-models")
async def recommend_models(
    file: UploadFile = File(...),
    target_column: str = Form(...)
):
    """
    Endpoint to upload a CSV file and get ML model recommendations.
    """
    # 1. Validate file extension
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only CSV files are allowed."
        )

    try:
        # 2. Read the uploaded file into a pandas DataFrame
        contents = await file.read()
        
        # We use io.BytesIO to treat the raw bytes as a file object
        try:
            df = pd.read_csv(io.BytesIO(contents))
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Could not parse the CSV file. Error: {str(e)}"
            )
            
        # Check if the dataframe is empty
        if df.empty:
            raise HTTPException(
                status_code=400,
                detail="The uploaded CSV file is empty."
            )

        # 3. Validate that the target column exists in the DataFrame
        if target_column not in df.columns:
            available_columns = list(df.columns)
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Target column not found.",
                    "provided_target": target_column,
                    "available_columns": available_columns
                }
            )

        # 4. Call the get_recommendations function
        result = get_recommendations(df, target_column)
        
        # Check if the result from the MLAdvisor module itself was an error
        if result.get("status") == "error":
            return JSONResponse(status_code=400, content=result)

        # 5. Return successful JSON response
        return JSONResponse(content=result)

    except HTTPException:
        # Re-raise HTTPExceptions so FastAPI can handle them normally
        raise
    except Exception as e:
        # 6. Global exception handler for unexpected errors during processing
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}"
        )
