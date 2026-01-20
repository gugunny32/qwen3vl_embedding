# Multimodal RAG with Qwen3-VL and Supabase

A FastAPI-based multimodal Retrieval-Augmented Generation (RAG) system using Qwen3-VL models, Supabase PostgreSQL, and hybrid search capabilities.

## 🎯 Project Overview

This project implements a production-ready multimodal RAG system that:
- Processes PDF documents with text and images
- Generates embeddings using Qwen3-VL-Embedding-2B (1024-dimensional)
- Performs hybrid search (semantic + full-text search)
- Re-ranks results using Qwen3-VL-Reranker-2B
- Generates Thai language responses using SeaLLM v3 model
- Stores data in Supabase PostgreSQL with vector support

## 🏗️ Architecture

```
┌─────────────┐
│   FastAPI   │
│   Backend   │
└─────┬───────┘
      │
      ├─── PDF Processing (PyMuPDF)
      │
      ├─── Qwen3-VL-Embedding (1024-dim)
      │
      ├─── Supabase PostgreSQL
      │    ├── documents table (metadata)
      │    └── chunks table (embeddings + text)
      │
      ├─── Hybrid Search
      │    ├── Semantic Search (pgvector)
      │    └── Full-Text Search (PostgreSQL)
      │
      ├─── Qwen3-VL-Reranker
      │
      └─── SeaLLM v3 (Thai Generation)
```

## 📋 Implementation Plan

### Phase 1: Infrastructure Setup ✅
- [x] Initialize project structure
- [x] Create Docker Compose with NVIDIA GPU support
- [x] Set up Supabase PostgreSQL connection
- [x] Create database schema (documents + chunks tables)
- [x] Configure pgvector extension

### Phase 2: Model Integration ⏳
- [ ] Integrate Qwen3-VL-Embedding-2B (1024-dim)
- [ ] Integrate Qwen3-VL-Reranker-2B
- [ ] Integrate SeaLLM v3 for Thai generation
- [ ] Create model management utilities
- [ ] Implement GPU memory optimization

### Phase 3: Document Processing 📄
- [ ] PDF text extraction
- [ ] PDF image extraction
- [ ] Intelligent chunking strategy
- [ ] Multimodal embedding generation
- [ ] Batch processing support

### Phase 4: Search Implementation 🔍
- [ ] Semantic search with pgvector
- [ ] Full-text search with PostgreSQL
- [ ] Hybrid search score fusion
- [ ] Re-ranking pipeline
- [ ] Result filtering and pagination

### Phase 5: API Development 🚀
- [ ] Upload document endpoint
- [ ] Search endpoint
- [ ] Query with RAG endpoint
- [ ] Document management endpoints
- [ ] Health check and status endpoints

### Phase 6: Testing & Documentation 📚
- [ ] Create test scripts
- [ ] Test with Thai PDF documents
- [ ] Performance benchmarking
- [ ] API documentation
- [ ] Deployment guide

## 📦 Technology Stack

### Backend
- **FastAPI**: Modern web framework
- **Python 3.10+**: Programming language
- **PyMuPDF (fitz)**: PDF processing
- **Transformers**: Model inference

### Models
- **Qwen3-VL-Embedding-2B**: Multimodal embeddings (1024-dim)
- **Qwen3-VL-Reranker-2B**: Result re-ranking
- **SeaLLM v3 (7B)**: Thai language generation

### Database
- **Supabase PostgreSQL**: Primary database
- **pgvector**: Vector similarity search
- **pg_trgm**: Full-text search

### DevOps
- **Docker & Docker Compose**: Containerization
- **NVIDIA Container Toolkit**: GPU support

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- NVIDIA GPU with drivers
- NVIDIA Container Toolkit
- 16GB+ GPU memory recommended

### Setup

1. Clone the repository:
```bash
git clone https://github.com/gugunny32/qwen3vl_embedding.git
cd qwen3vl_embedding
```

2. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your Supabase credentials
```

3. Start services:
```bash
docker-compose up -d
```

4. Initialize database:
```bash
docker-compose exec api python scripts/init_db.py
```

5. Test the API:
```bash
curl http://localhost:8000/health
```

## 📁 Project Structure

```
multimodal_embedding/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application
│   ├── config.py               # Configuration
│   ├── models/
│   │   ├── __init__.py
│   │   ├── embedding.py        # Qwen3-VL-Embedding
│   │   ├── reranker.py         # Qwen3-VL-Reranker
│   │   └── generator.py        # SeaLLM v3
│   ├── services/
│   │   ├── __init__.py
│   │   ├── pdf_processor.py    # PDF processing
│   │   ├── chunker.py          # Text chunking
│   │   ├── search.py           # Hybrid search
│   │   └── rag.py              # RAG pipeline
│   ├── database/
│   │   ├── __init__.py
│   │   ├── connection.py       # Supabase connection
│   │   ├── schema.sql          # Database schema
│   │   └── operations.py       # CRUD operations
│   └── api/
│       ├── __init__.py
│       ├── documents.py        # Document endpoints
│       ├── search.py           # Search endpoints
│       └── rag.py              # RAG endpoints
├── scripts/
│   ├── init_db.py              # Database initialization
│   └── test_api.py             # API testing
├── test_file/
│   └── pdf/                    # Test PDF files
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## 🔧 Configuration

### Environment Variables

```env
# Supabase Configuration
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
SUPABASE_DB_URL=postgresql://user:pass@host:5432/db

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
HYBRID_ALPHA=0.5  # 0.5 = equal weight to semantic and text search
```

## 📝 API Endpoints

### Document Management

#### Upload Document
```bash
POST /api/v1/documents/upload
Content-Type: multipart/form-data

curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -F "file=@document.pdf" \
  -F "metadata={\"title\":\"Document Title\"}"
```

#### List Documents
```bash
GET /api/v1/documents
```

#### Get Document
```bash
GET /api/v1/documents/{document_id}
```

#### Delete Document
```bash
DELETE /api/v1/documents/{document_id}
```

### Search

#### Hybrid Search
```bash
POST /api/v1/search
Content-Type: application/json

{
  "query": "ขั้นตอนการจัดซื้อคืออะไร",
  "top_k": 5,
  "use_reranker": true
}
```

### RAG

#### Query with RAG
```bash
POST /api/v1/rag/query
Content-Type: application/json

{
  "question": "อธิบายขั้นตอนการจัดซื้อในระบบ",
  "top_k": 5
}
```

## 🎯 Why 1024 Dimensions?

We chose 1024-dimensional embeddings as a balance between:
- ✅ **Performance**: Faster similarity search vs 2048-dim
- ✅ **Storage**: 50% less disk space and memory
- ✅ **Quality**: Still captures semantic meaning effectively
- ✅ **Cost**: Lower computational requirements

For most RAG applications, 1024 dimensions provide excellent results while maintaining efficiency.

## 🧪 Testing

Run the test suite:
```bash
# Test document upload
python scripts/test_api.py --test upload

# Test search
python scripts/test_api.py --test search

# Test RAG
python scripts/test_api.py --test rag

# Full test suite
python scripts/test_api.py --all
```

## 📊 Database Schema

### Documents Table
```sql
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename VARCHAR(255) NOT NULL,
    title TEXT,
    metadata JSONB,
    total_chunks INTEGER,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

### Chunks Table
```sql
CREATE TABLE chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR(1024),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW(),

    -- Full-text search
    content_tsv TSVECTOR GENERATED ALWAYS AS (to_tsvector('thai', content)) STORED
);

-- Indexes
CREATE INDEX idx_chunks_document_id ON chunks(document_id);
CREATE INDEX idx_chunks_embedding ON chunks USING ivfflat (embedding vector_cosine_ops);
CREATE INDEX idx_chunks_content_tsv ON chunks USING gin(content_tsv);
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

MIT License

## 🙏 Acknowledgments

- [Qwen3-VL Models](https://huggingface.co/Qwen) by Alibaba Cloud
- [SeaLLM](https://huggingface.co/SeaLLMs) for Thai language support
- [Supabase](https://supabase.com/) for managed PostgreSQL
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework

## 📧 Contact

For questions or support, please open an issue on GitHub.
