-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Documents table
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename VARCHAR(255) NOT NULL,
    title TEXT,
    metadata JSONB DEFAULT '{}',
    total_chunks INTEGER DEFAULT 0,
    file_size BIGINT,
    content_hash VARCHAR(64),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index on filename
CREATE INDEX IF NOT EXISTS idx_documents_filename ON documents(filename);
CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at DESC);

-- Chunks table
CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR(1024),
    metadata JSONB DEFAULT '{}',
    has_image BOOLEAN DEFAULT FALSE,
    image_path TEXT,
    page_number INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Ensure unique chunk per document
    UNIQUE(document_id, chunk_index)
);

-- Full-text search column (generated)
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS content_tsv TSVECTOR
    GENERATED ALWAYS AS (to_tsvector('simple', content)) STORED;

-- Create indexes for chunks
CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_chunk_index ON chunks(document_id, chunk_index);

-- Vector similarity search index (IVFFlat for faster approximate search)
-- Note: This requires at least 1000 rows for training. Will be created after initial data insertion.
-- CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks
--     USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- For now, use a basic index
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks USING hnsw (embedding vector_cosine_ops);

-- Full-text search index
CREATE INDEX IF NOT EXISTS idx_chunks_content_tsv ON chunks USING gin(content_tsv);

-- Page number index for filtering
CREATE INDEX IF NOT EXISTS idx_chunks_page_number ON chunks(page_number);

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for documents table
DROP TRIGGER IF EXISTS update_documents_updated_at ON documents;
CREATE TRIGGER update_documents_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Function for hybrid search
CREATE OR REPLACE FUNCTION hybrid_search(
    query_embedding VECTOR(1024),
    query_text TEXT,
    match_limit INTEGER DEFAULT 10,
    semantic_weight FLOAT DEFAULT 0.5
)
RETURNS TABLE (
    chunk_id UUID,
    document_id UUID,
    content TEXT,
    metadata JSONB,
    page_number INTEGER,
    semantic_score FLOAT,
    text_score FLOAT,
    hybrid_score FLOAT
) AS $$
BEGIN
    RETURN QUERY
    WITH semantic_search AS (
        SELECT
            c.id,
            c.document_id,
            c.content,
            c.metadata,
            c.page_number,
            1 - (c.embedding <=> query_embedding) AS score
        FROM chunks c
        WHERE c.embedding IS NOT NULL
        ORDER BY c.embedding <=> query_embedding
        LIMIT match_limit * 2
    ),
    text_search AS (
        SELECT
            c.id,
            c.document_id,
            c.content,
            c.metadata,
            c.page_number,
            ts_rank(c.content_tsv, plainto_tsquery('simple', query_text)) AS score
        FROM chunks c
        WHERE c.content_tsv @@ plainto_tsquery('simple', query_text)
        ORDER BY score DESC
        LIMIT match_limit * 2
    )
    SELECT
        COALESCE(s.id, t.id) AS chunk_id,
        COALESCE(s.document_id, t.document_id) AS document_id,
        COALESCE(s.content, t.content) AS content,
        COALESCE(s.metadata, t.metadata) AS metadata,
        COALESCE(s.page_number, t.page_number) AS page_number,
        COALESCE(s.score, 0.0) AS semantic_score,
        COALESCE(t.score, 0.0) AS text_score,
        (COALESCE(s.score, 0.0) * semantic_weight + COALESCE(t.score, 0.0) * (1 - semantic_weight)) AS hybrid_score
    FROM semantic_search s
    FULL OUTER JOIN text_search t ON s.id = t.id
    ORDER BY hybrid_score DESC
    LIMIT match_limit;
END;
$$ LANGUAGE plpgsql;

-- Comments for documentation
COMMENT ON TABLE documents IS 'Stores document metadata';
COMMENT ON TABLE chunks IS 'Stores document chunks with embeddings for RAG';
COMMENT ON COLUMN chunks.embedding IS 'Qwen3-VL-Embedding-2B generated embedding (1024 dimensions)';
COMMENT ON COLUMN chunks.content_tsv IS 'Full-text search vector, auto-generated from content';
COMMENT ON FUNCTION hybrid_search IS 'Performs hybrid search combining semantic and full-text search';
