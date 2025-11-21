# ===========================================
# AI-Powered Recommendation System
# Multi-stage Production Dockerfile
# ===========================================

# Build stage - install dependencies
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip wheel setuptools && \
    pip install --no-cache-dir -r requirements.txt

# Copy and install the package
COPY pyproject.toml .
COPY src/ /app/src/
RUN pip install --no-cache-dir .

# ===========================================
# Production stage
# ===========================================
FROM python:3.11-slim as production

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --chown=appuser:appuser src/ /app/src/

# Create directories for models, data, and logs
RUN mkdir -p /app/models /app/data /app/logs && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    APP_ENV=production \
    HOST=0.0.0.0 \
    PORT=8000

# Expose port
EXPOSE 8000

# Health check using curl (more reliable than Python in production)
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health/live || exit 1

# Run the application with Gunicorn for production
CMD ["python", "-m", "uvicorn", "recommendation_system.api.app:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "4", "--loop", "uvloop", "--http", "httptools"]

# ===========================================
# Development stage (optional)
# ===========================================
FROM production as development

USER root

# Install development dependencies
RUN pip install --no-cache-dir pytest pytest-cov pytest-asyncio black ruff mypy

USER appuser

ENV APP_ENV=development \
    DEBUG=true

CMD ["python", "-m", "uvicorn", "recommendation_system.api.app:app", \
     "--host", "0.0.0.0", "--port", "8000", "--reload"]
