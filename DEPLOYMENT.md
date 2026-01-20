# Deployment Guide

This guide provides detailed instructions for deploying the Multimodal RAG system.

## Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA GPU with at least 16GB VRAM (24GB recommended)
- **RAM**: 32GB+ system RAM
- **Storage**: 100GB+ free space for models and data
- **CPU**: 8+ cores recommended

### Software Requirements
- Ubuntu 20.04+ or similar Linux distribution
- Docker 20.10+
- Docker Compose 2.0+
- NVIDIA Driver 525+ (for CUDA 12.1)
- NVIDIA Container Toolkit

## Installation Steps

### 1. Install NVIDIA Drivers

```bash
# Check current driver version
nvidia-smi

# If needed, install/update NVIDIA drivers
sudo apt update
sudo apt install nvidia-driver-525
sudo reboot
```

### 2. Install Docker

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Verify installation
docker --version
```

### 3. Install Docker Compose

```bash
# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Verify installation
docker-compose --version
```

### 4. Install NVIDIA Container Toolkit

```bash
# Add NVIDIA Container Toolkit repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker to use NVIDIA runtime
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# Verify GPU access
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

## Setup Supabase

### Option 1: Supabase Cloud (Recommended for Production)

1. Create account at [supabase.com](https://supabase.com)
2. Create a new project
3. Get your connection details:
   - Project URL: `https://[project-ref].supabase.co`
   - Anon Key: From Settings > API
   - Database URL: From Settings > Database

### Option 2: Self-Hosted Supabase

```bash
# Clone Supabase
git clone --depth 1 https://github.com/supabase/supabase

# Setup
cd supabase/docker
cp .env.example .env

# Edit .env with your settings
nano .env

# Start Supabase
docker-compose up -d

# Access at http://localhost:8000
```

## Deploy Application

### 1. Clone Repository

```bash
git clone https://github.com/gugunny32/qwen3vl_embedding.git
cd qwen3vl_embedding
```

### 2. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit with your Supabase credentials
nano .env
```

Edit `.env`:
```env
# Supabase Configuration
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key
SUPABASE_DB_URL=postgresql://postgres:your-password@db.your-project.supabase.co:5432/postgres

# Model Configuration
EMBEDDING_MODEL=Qwen/Qwen3-VL-Embedding-2B
EMBEDDING_DIMENSION=1024
RERANKER_MODEL=Qwen/Qwen3-VL-Reranker-2B
GENERATOR_MODEL=SeaLLMs/SeaLLM-7B-v3

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
MAX_CHUNK_SIZE=512
CHUNK_OVERLAP=50

# Search Configuration
TOP_K=20
RERANK_TOP_K=5
HYBRID_ALPHA=0.5

# GPU Configuration
CUDA_VISIBLE_DEVICES=0
```

### 3. Build and Start Services

```bash
# Build Docker image
docker-compose build

# Start services
docker-compose up -d

# Check logs
docker-compose logs -f api
```

### 4. Initialize Database

```bash
# Run database initialization
docker-compose exec api python scripts/init_db.py

# Verify tables were created
# You should see: documents, chunks tables and extensions
```

### 5. Verify Deployment

```bash
# Check health
curl http://localhost:8000/health

# Check status
curl http://localhost:8000/status

# Expected response:
# {
#   "status": "healthy",
#   "service": "multimodal-rag-api",
#   "database": "connected",
#   "cuda_available": true
# }
```

## Testing

### Upload Test Document

```bash
# Upload a PDF
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -F "file=@test_file/pdf/FAQ.pdf" \
  -F "title=FAQ Document"
```

### Test Search

```bash
# Search
curl -X POST "http://localhost:8000/api/v1/search/hybrid" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "ระบบจัดซื้อ",
    "top_k": 5,
    "use_reranker": true
  }'
```

### Test RAG

```bash
# Ask a question
curl -X POST "http://localhost:8000/api/v1/rag/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "อธิบายขั้นตอนการจัดซื้อ",
    "top_k": 5,
    "use_reranker": true
  }'
```

### Run Full Test Suite

```bash
# Run all tests
python scripts/test_api.py --test all

# Or test individual components
python scripts/test_api.py --test health
python scripts/test_api.py --test upload --pdf test_file/pdf/FAQ.pdf
python scripts/test_api.py --test search --query "ระบบจัดซื้อ"
python scripts/test_api.py --test rag --question "อธิบายขั้นตอนการจัดซื้อ"
```

## Production Considerations

### 1. Security

```bash
# Use strong database passwords
# Restrict CORS origins in app/main.py
# Use HTTPS with reverse proxy (nginx/traefik)
# Add authentication middleware
# Rate limiting
```

### 2. Reverse Proxy (Nginx)

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Increase timeout for long-running requests
        proxy_read_timeout 300s;
        proxy_connect_timeout 75s;
    }
}
```

### 3. SSL/TLS with Let's Encrypt

```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx

# Get certificate
sudo certbot --nginx -d your-domain.com

# Auto-renewal is configured automatically
```

### 4. Monitoring

```bash
# View logs
docker-compose logs -f api

# Check resource usage
docker stats

# Monitor GPU
watch -n 1 nvidia-smi
```

### 5. Backup

```bash
# Backup database
pg_dump -h db.your-project.supabase.co -U postgres -d postgres > backup.sql

# Restore database
psql -h db.your-project.supabase.co -U postgres -d postgres < backup.sql

# Backup uploaded files
tar -czf uploads_backup.tar.gz uploads/
```

### 6. Scaling

For production workloads:

1. **Horizontal Scaling**: Run multiple API instances behind a load balancer
2. **Database**: Use Supabase Cloud with connection pooling
3. **Model Caching**: Use persistent volumes for model cache
4. **Queue System**: Add Celery/RabbitMQ for async processing

### 7. Performance Optimization

```yaml
# docker-compose.yml optimizations
services:
  api:
    deploy:
      resources:
        limits:
          memory: 32G
        reservations:
          memory: 16G
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      # Optimize model loading
      - TRANSFORMERS_CACHE=/app/models
      - HF_HOME=/app/models
      # Optimize GPU memory
      - PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

## Troubleshooting

### GPU Not Detected

```bash
# Check NVIDIA driver
nvidia-smi

# Check Docker GPU access
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# Restart Docker
sudo systemctl restart docker
```

### Out of Memory Errors

```bash
# Reduce batch size in config
# Use model quantization (already enabled)
# Use smaller models
# Increase GPU memory allocation
```

### Database Connection Issues

```bash
# Test connection
psql -h db.your-project.supabase.co -U postgres -d postgres

# Check firewall rules
# Verify credentials in .env
# Check Supabase dashboard for connection limits
```

### Model Download Issues

```bash
# Check internet connection
# Use Hugging Face token if needed
export HF_TOKEN=your_token

# Manual download
from transformers import AutoModel
model = AutoModel.from_pretrained("Qwen/Qwen3-VL-Embedding-2B")
```

## Maintenance

### Update Models

```bash
# Clear model cache
docker-compose exec api rm -rf /root/.cache/huggingface

# Restart to download latest models
docker-compose restart api
```

### Update Application

```bash
# Pull latest code
git pull origin main

# Rebuild and restart
docker-compose down
docker-compose build
docker-compose up -d
```

### Clean Up

```bash
# Remove unused images
docker system prune -a

# Remove old containers
docker-compose down --volumes

# Clean model cache
rm -rf ~/.cache/huggingface
```

## Support

For issues and questions:
- GitHub Issues: https://github.com/gugunny32/qwen3vl_embedding/issues
- Check logs: `docker-compose logs -f api`
- Monitor GPU: `nvidia-smi -l 1`
