# ================================
# Stage 1: Build frontend
# ================================
FROM node:18-alpine AS frontend-builder
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm install
COPY frontend/ .
RUN npm run build

# ================================
# Stage 2: Build backend
# ================================
FROM python:3.11-slim
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# Copy backend code and requirements
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/ .

# Copy built frontend into backend's static folder (if FastAPI serves frontend)
COPY --from=frontend-builder /frontend/build ./static

# Expose backend port
EXPOSE 8080

# Start FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
