# Immersion Cooling Validator - Docker Image
# Build: docker build -t icv-web .
# Run: docker run -p 8000:8000 icv-web

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY web/backend/requirements.txt requirements-web.txt
COPY pyproject.toml .
RUN pip install --no-cache-dir -r requirements-web.txt

# Copy source code
COPY src/ src/
COPY web/ web/

# Install ICV package
RUN pip install --no-cache-dir -e .

# Set working directory to backend
WORKDIR /app/web/backend

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
