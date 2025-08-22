# Use slim Python image
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1
    
# Set working directory
WORKDIR /app

# Install system dependencies (optional, e.g. for FAISS, PDFs, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first (better caching)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Run with uvicorn (no reload in production)
# CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
CMD ["gunicorn", "main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--capture-output", "--access-logfile", "-", "--error-logfile", "-", "--log-level", "info"]
