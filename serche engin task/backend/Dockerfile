FROM python:3.11-slim

WORKDIR /app

# Install system build dependencies (e.g. for package compilations)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker layer caching
COPY requirements.txt .

# Install dependencies (utilizing pip cache)
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the sentence-transformers model to speed up container boot
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')"

# Copy application source code
COPY . .

# Expose port
EXPOSE 8000

# Environment variables defaults
ENV PYTHONUNBUFFERED=1
ENV HOST=0.0.0.0
ENV PORT=8000

# Run Uvicorn server
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
