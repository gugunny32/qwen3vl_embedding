# API Documentation

Complete API reference for the Multimodal RAG system.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, the API does not require authentication. In production, consider adding:
- JWT tokens
- API keys
- OAuth 2.0

## Response Format

All responses are in JSON format.

### Success Response
```json
{
  "data": {...},
  "status": "success"
}
```

### Error Response
```json
{
  "detail": "Error message",
  "status": "error"
}
```

---

## Health & Status Endpoints

### Health Check
Check if the API is running.

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "service": "multimodal-rag-api",
  "version": "1.0.0",
  "database": "connected"
}
```

### Status
Get detailed system status.

**Endpoint:** `GET /status`

**Response:**
```json
{
  "service": "multimodal-rag-api",
  "version": "1.0.0",
  "status": "running",
  "cuda_available": true,
  "gpu_count": 1,
  "gpu_name": "NVIDIA RTX 4090",
  "settings": {
    "embedding_model": "Qwen/Qwen3-VL-Embedding-2B",
    "embedding_dimension": 1024,
    "reranker_model": "Qwen/Qwen3-VL-Reranker-2B",
    "generator_model": "SeaLLMs/SeaLLM-7B-v3",
    "max_chunk_size": 512,
    "top_k": 20,
    "hybrid_alpha": 0.5
  }
}
```

---

## Document Management

### Upload Document

Upload a PDF document for processing.

**Endpoint:** `POST /api/v1/documents/upload`

**Content-Type:** `multipart/form-data`

**Parameters:**
- `file` (required): PDF file
- `title` (optional): Document title

**Example:**
```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -F "file=@document.pdf" \
  -F "title=My Document"
```

**Response:**
```json
{
  "document_id": "processing",
  "filename": "document.pdf",
  "status": "processing",
  "message": "Document is being processed in the background",
  "total_chunks": 0
}
```

**Note:** Document processing happens asynchronously. Check document status with GET request.

---

### List Documents

Get all uploaded documents.

**Endpoint:** `GET /api/v1/documents/`

**Query Parameters:**
- `limit` (optional): Number of documents (default: 100)
- `offset` (optional): Offset for pagination (default: 0)

**Example:**
```bash
curl "http://localhost:8000/api/v1/documents/?limit=10&offset=0"
```

**Response:**
```json
[
  {
    "id": "123e4567-e89b-12d3-a456-426614174000",
    "filename": "document.pdf",
    "title": "My Document",
    "total_chunks": 25,
    "file_size": 1024576,
    "created_at": "2024-01-20T10:30:00Z"
  }
]
```

---

### Get Document

Get details of a specific document.

**Endpoint:** `GET /api/v1/documents/{document_id}`

**Example:**
```bash
curl "http://localhost:8000/api/v1/documents/123e4567-e89b-12d3-a456-426614174000"
```

**Response:**
```json
{
  "id": "123e4567-e89b-12d3-a456-426614174000",
  "filename": "document.pdf",
  "title": "My Document",
  "total_chunks": 25,
  "file_size": 1024576,
  "created_at": "2024-01-20T10:30:00Z"
}
```

---

### Get Document Chunks

Get chunks for a specific document.

**Endpoint:** `GET /api/v1/documents/{document_id}/chunks`

**Query Parameters:**
- `limit` (optional): Number of chunks (default: 100)
- `offset` (optional): Offset for pagination (default: 0)

**Example:**
```bash
curl "http://localhost:8000/api/v1/documents/123e4567-e89b-12d3-a456-426614174000/chunks"
```

**Response:**
```json
[
  {
    "id": "chunk-id-1",
    "chunk_index": 0,
    "content": "This is the first chunk...",
    "page_number": 1,
    "has_image": false
  }
]
```

---

### Delete Document

Delete a document and all its chunks.

**Endpoint:** `DELETE /api/v1/documents/{document_id}`

**Example:**
```bash
curl -X DELETE "http://localhost:8000/api/v1/documents/123e4567-e89b-12d3-a456-426614174000"
```

**Response:**
```json
{
  "message": "Document deleted successfully",
  "document_id": "123e4567-e89b-12d3-a456-426614174000"
}
```

---

## Search Endpoints

### Semantic Search

Search using embeddings only.

**Endpoint:** `POST /api/v1/search/semantic`

**Request Body:**
```json
{
  "query": "ระบบจัดซื้อ",
  "top_k": 5,
  "document_id": "optional-document-id"
}
```

**Response:**
```json
{
  "query": "ระบบจัดซื้อ",
  "results": [
    {
      "chunk_id": "chunk-1",
      "document_id": "doc-1",
      "content": "ระบบจัดซื้อเป็นระบบที่...",
      "page_number": 5,
      "similarity_score": 0.92,
      "text_score": null,
      "hybrid_score": null,
      "rerank_score": null
    }
  ],
  "num_results": 5,
  "search_type": "semantic"
}
```

---

### Text Search

Search using full-text search only.

**Endpoint:** `POST /api/v1/search/text`

**Request Body:**
```json
{
  "query": "ระบบจัดซื้อ",
  "top_k": 5
}
```

**Response:** Similar to semantic search with `text_score` populated.

---

### Hybrid Search

Search using both semantic and text search with optional reranking.

**Endpoint:** `POST /api/v1/search/hybrid`

**Request Body:**
```json
{
  "query": "ระบบจัดซื้อ",
  "top_k": 5,
  "use_reranker": true,
  "semantic_weight": 0.5,
  "document_id": null
}
```

**Parameters:**
- `query` (required): Search query
- `top_k` (optional): Number of results (default: 10, max: 100)
- `use_reranker` (optional): Apply reranking (default: false)
- `semantic_weight` (optional): Weight for semantic search 0-1 (default: 0.5)
- `document_id` (optional): Search within specific document

**Response:**
```json
{
  "query": "ระบบจัดซื้อ",
  "results": [
    {
      "chunk_id": "chunk-1",
      "document_id": "doc-1",
      "content": "ระบบจัดซื้อเป็นระบบที่...",
      "page_number": 5,
      "similarity_score": 0.85,
      "text_score": 0.72,
      "hybrid_score": 0.785,
      "rerank_score": 0.95
    }
  ],
  "num_results": 5,
  "search_type": "hybrid_with_rerank"
}
```

---

### Default Search

Uses hybrid search by default.

**Endpoint:** `POST /api/v1/search/`

Same as hybrid search endpoint.

---

## RAG (Retrieval-Augmented Generation)

### Query with RAG

Ask a question and get an AI-generated answer based on retrieved contexts.

**Endpoint:** `POST /api/v1/rag/query`

**Request Body:**
```json
{
  "question": "อธิบายขั้นตอนการจัดซื้อในระบบ",
  "top_k": 5,
  "use_reranker": true,
  "document_id": null,
  "temperature": 0.7,
  "max_tokens": 512
}
```

**Parameters:**
- `question` (required): User's question
- `top_k` (optional): Number of contexts to retrieve (default: 5, max: 20)
- `use_reranker` (optional): Use reranker (default: true)
- `document_id` (optional): Search within specific document
- `temperature` (optional): Generation temperature 0-2 (default: 0.7)
- `max_tokens` (optional): Max tokens to generate (default: 512, max: 2048)

**Example:**
```bash
curl -X POST "http://localhost:8000/api/v1/rag/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "อธิบายขั้นตอนการจัดซื้อ",
    "top_k": 5,
    "use_reranker": true
  }'
```

**Response:**
```json
{
  "question": "อธิบายขั้นตอนการจัดซื้อ",
  "answer": "ขั้นตอนการจัดซื้อในระบบประกอบด้วย...",
  "contexts": [
    "บริบทที่ 1: ระบบจัดซื้อเริ่มต้นจาก...",
    "บริบทที่ 2: ขั้นตอนถัดไปคือ..."
  ],
  "source_documents": [
    {
      "document_id": "doc-1",
      "page_number": 5,
      "chunk_id": "chunk-1",
      "score": 0.95,
      "content_preview": "ระบบจัดซื้อเป็นระบบที่..."
    }
  ],
  "num_sources": 2
}
```

---

### Query with Conversation

Query with conversation history for multi-turn conversations.

**Endpoint:** `POST /api/v1/rag/query/conversation`

**Request Body:**
```json
{
  "question": "แล้วขั้นตอนถัดไปล่ะ",
  "conversation_history": [
    {
      "role": "user",
      "content": "อธิบายขั้นตอนการจัดซื้อ"
    },
    {
      "role": "assistant",
      "content": "ขั้นตอนการจัดซื้อมี 3 ขั้นตอน..."
    }
  ],
  "top_k": 5,
  "use_reranker": true
}
```

**Response:** Same as regular query endpoint.

---

### Query with Citations

Get answer with inline citations to source documents.

**Endpoint:** `POST /api/v1/rag/query/citations`

**Request Body:** Same as regular query endpoint.

**Response:**
```json
{
  "question": "อธิบายขั้นตอนการจัดซื้อ",
  "answer": "ขั้นตอนการจัดซื้อประกอบด้วย...\n\n**แหล่งอ้างอิง:**\n1. เอกสาร 123e4567... หน้า 5\n2. เอกสาร 789abcde... หน้า 12",
  "contexts": [...],
  "source_documents": [...],
  "num_sources": 2
}
```

---

### Summarize Document

Generate a summary of a document.

**Endpoint:** `POST /api/v1/rag/summarize`

**Request Body:**
```json
{
  "document_id": "123e4567-e89b-12d3-a456-426614174000",
  "max_tokens": 512
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/api/v1/rag/summarize" \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "123e4567-e89b-12d3-a456-426614174000",
    "max_tokens": 512
  }'
```

**Response:**
```json
{
  "document_id": "123e4567-e89b-12d3-a456-426614174000",
  "summary": "เอกสารนี้กล่าวถึง...",
  "num_chunks": 25
}
```

---

## Error Codes

| Code | Description |
|------|-------------|
| 200  | Success |
| 400  | Bad Request - Invalid parameters |
| 404  | Not Found - Resource doesn't exist |
| 500  | Internal Server Error |
| 503  | Service Unavailable - Health check failed |

---

## Rate Limiting

Currently no rate limiting. Consider implementing for production:
- Per IP: 100 requests/minute
- Per endpoint: Varies by complexity

---

## Best Practices

### 1. Document Upload
- Upload PDFs one at a time
- Wait for processing to complete before querying
- Check document chunks count to verify processing

### 2. Search
- Use hybrid search for best results
- Enable reranker for improved accuracy (slower)
- Adjust `semantic_weight` based on query type:
  - 0.7-0.8: Semantic queries (concepts, meanings)
  - 0.3-0.5: Keyword queries (specific terms, names)

### 3. RAG Queries
- Use `top_k=5` for focused answers
- Use `top_k=10` for comprehensive answers
- Adjust temperature:
  - 0.3-0.5: Factual, precise answers
  - 0.7-0.9: Creative, detailed answers

### 4. Performance
- Batch upload documents during off-peak hours
- Cache frequently accessed documents
- Use document_id filter when possible

---

## Examples

### Complete Workflow

```bash
# 1. Check health
curl http://localhost:8000/health

# 2. Upload document
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -F "file=@document.pdf" \
  -F "title=Important Document"

# 3. Wait for processing (check periodically)
sleep 30

# 4. List documents to get ID
curl http://localhost:8000/api/v1/documents/

# 5. Search
curl -X POST "http://localhost:8000/api/v1/search/hybrid" \
  -H "Content-Type: application/json" \
  -d '{"query": "ระบบจัดซื้อ", "top_k": 5, "use_reranker": true}'

# 6. Ask question
curl -X POST "http://localhost:8000/api/v1/rag/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "อธิบายระบบจัดซื้อ", "top_k": 5}'
```

---

## Interactive Documentation

Access interactive API documentation at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

These provide:
- Interactive API testing
- Request/response examples
- Schema definitions
- Try-it-out functionality
