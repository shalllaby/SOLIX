# ==========================================
# STAGE 1: Builder (Dependency installation)
# ==========================================
FROM python:3.10-slim AS builder

WORKDIR /app

# Install compilation tools needed for potential source builds
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment to isolate dependencies cleanly
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ==========================================
# STAGE 2: Runner (Minimal production image)
# ==========================================
FROM python:3.10-slim AS runner

WORKDIR /app

# Install system dependencies
# - tesseract-ocr & tesseract-ocr-eng for PyTesseract OCR subsystem
# - libgl1-mesa-glx & libglib2.0-0 for OpenCV / PDF generation rendering libraries
# - sqlite3 for database backup operations and CLI access
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    libglib2.0-0 \
    libgl1-mesa-glx \
    sqlite3 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set environment variables for clean logs and execution
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Copy code structure (only what is required)
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY core/ ./core/
COPY utils/ ./utils/
COPY data_layer/ ./data_layer/
COPY SOLIX_DOCUMENTATION.md .

# Copy default team images
COPY ["صور التيم/", "./صور التيم/"]

# CRITICAL FIX: Ensure directories mounted by FastAPI exist to prevent startup failure
# e.g., FastAPI fails to mount directories if they are missing
RUN mkdir -p "صور التيم" backend/data

EXPOSE 8000

# Start server using Uvicorn CLI to allow binding to 0.0.0.0
# We run 4 workers to handle high-concurrency requests safely
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
