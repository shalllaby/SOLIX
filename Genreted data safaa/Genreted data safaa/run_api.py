from fastapi import FastAPI
import json
import uvicorn

app = FastAPI(title="Synthetic Data API", description="Auto-generated API for Synthetic Data")

# تحميل البيانات
with open("mock_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

@app.get("/")
def home():
    return {"message": "Welcome to the Synthetic Data API!", "total_records": len(data)}

@app.get("/data")
def get_data(limit: int = 100, skip: int = 0):
    return data[skip : skip + limit]

@app.get("/data/{id}")
def get_single(id: int):
    if id < len(data):
        return data[id]
    return {"error": "Record not found"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
