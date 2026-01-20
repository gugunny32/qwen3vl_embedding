from typing import List, Dict, Any, Optional, Tuple
from uuid import UUID
import hashlib
from loguru import logger
import numpy as np

from app.database.connection import get_database


class DocumentOperations:
    def __init__(self):
        self.db = get_database()

    def create_document(
        self,
        filename: str,
        title: Optional[str] = None,
        metadata: Optional[Dict] = None,
        file_size: Optional[int] = None,
        content_hash: Optional[str] = None
    ) -> UUID:
        """Create a new document record"""
        query = """
            INSERT INTO documents (filename, title, metadata, file_size, content_hash)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
        """
        result = self.db.execute_query(
            query,
            (filename, title, metadata or {}, file_size, content_hash)
        )
        doc_id = result[0]['id']
        logger.info(f"Created document {doc_id}: {filename}")
        return doc_id

    def get_document(self, document_id: UUID) -> Optional[Dict]:
        """Get document by ID"""
        query = "SELECT * FROM documents WHERE id = %s"
        result = self.db.execute_query(query, (str(document_id),))
        return result[0] if result else None

    def get_all_documents(self, limit: int = 100, offset: int = 0) -> List[Dict]:
        """Get all documents with pagination"""
        query = """
            SELECT * FROM documents
            ORDER BY created_at DESC
            LIMIT %s OFFSET %s
        """
        return self.db.execute_query(query, (limit, offset))

    def update_document_chunks_count(self, document_id: UUID, count: int):
        """Update total chunks count for a document"""
        query = "UPDATE documents SET total_chunks = %s WHERE id = %s"
        self.db.execute_query(query, (count, str(document_id)))

    def delete_document(self, document_id: UUID):
        """Delete a document (cascades to chunks)"""
        query = "DELETE FROM documents WHERE id = %s"
        self.db.execute_query(query, (str(document_id),))
        logger.info(f"Deleted document {document_id}")

    def document_exists_by_hash(self, content_hash: str) -> Optional[UUID]:
        """Check if document with same hash exists"""
        query = "SELECT id FROM documents WHERE content_hash = %s"
        result = self.db.execute_query(query, (content_hash,))
        return result[0]['id'] if result else None


class ChunkOperations:
    def __init__(self):
        self.db = get_database()

    def create_chunk(
        self,
        document_id: UUID,
        chunk_index: int,
        content: str,
        embedding: Optional[np.ndarray] = None,
        metadata: Optional[Dict] = None,
        has_image: bool = False,
        image_path: Optional[str] = None,
        page_number: Optional[int] = None
    ) -> UUID:
        """Create a new chunk"""
        query = """
            INSERT INTO chunks (
                document_id, chunk_index, content, embedding,
                metadata, has_image, image_path, page_number
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """

        # Convert numpy array to list for PostgreSQL
        embedding_list = embedding.tolist() if embedding is not None else None

        result = self.db.execute_query(
            query,
            (
                str(document_id), chunk_index, content, embedding_list,
                metadata or {}, has_image, image_path, page_number
            )
        )
        return result[0]['id']

    def create_chunks_batch(self, chunks_data: List[Dict]) -> List[UUID]:
        """Create multiple chunks in a batch"""
        query = """
            INSERT INTO chunks (
                document_id, chunk_index, content, embedding,
                metadata, has_image, image_path, page_number
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        """

        params_list = []
        for chunk in chunks_data:
            embedding = chunk.get('embedding')
            embedding_list = embedding.tolist() if embedding is not None else None

            params_list.append((
                str(chunk['document_id']),
                chunk['chunk_index'],
                chunk['content'],
                embedding_list,
                chunk.get('metadata', {}),
                chunk.get('has_image', False),
                chunk.get('image_path'),
                chunk.get('page_number')
            ))

        chunk_ids = []
        with self.db.get_cursor() as cursor:
            for params in params_list:
                cursor.execute(query, params)
                chunk_ids.append(cursor.fetchone()['id'])

        logger.info(f"Created {len(chunk_ids)} chunks in batch")
        return chunk_ids

    def get_chunks_by_document(
        self,
        document_id: UUID,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict]:
        """Get all chunks for a document"""
        query = """
            SELECT id, chunk_index, content, metadata, page_number, has_image
            FROM chunks
            WHERE document_id = %s
            ORDER BY chunk_index
            LIMIT %s OFFSET %s
        """
        return self.db.execute_query(query, (str(document_id), limit, offset))

    def semantic_search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        document_id: Optional[UUID] = None
    ) -> List[Dict]:
        """Perform semantic search using vector similarity"""
        embedding_list = query_embedding.tolist()

        if document_id:
            query = """
                SELECT
                    id, document_id, content, metadata, page_number,
                    1 - (embedding <=> %s::vector) AS similarity
                FROM chunks
                WHERE embedding IS NOT NULL AND document_id = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """
            params = (embedding_list, str(document_id), embedding_list, top_k)
        else:
            query = """
                SELECT
                    id, document_id, content, metadata, page_number,
                    1 - (embedding <=> %s::vector) AS similarity
                FROM chunks
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """
            params = (embedding_list, embedding_list, top_k)

        return self.db.execute_query(query, params)

    def text_search(
        self,
        query_text: str,
        top_k: int = 10,
        document_id: Optional[UUID] = None
    ) -> List[Dict]:
        """Perform full-text search"""
        if document_id:
            query = """
                SELECT
                    id, document_id, content, metadata, page_number,
                    ts_rank(content_tsv, plainto_tsquery('simple', %s)) AS rank
                FROM chunks
                WHERE content_tsv @@ plainto_tsquery('simple', %s)
                    AND document_id = %s
                ORDER BY rank DESC
                LIMIT %s
            """
            params = (query_text, query_text, str(document_id), top_k)
        else:
            query = """
                SELECT
                    id, document_id, content, metadata, page_number,
                    ts_rank(content_tsv, plainto_tsquery('simple', %s)) AS rank
                FROM chunks
                WHERE content_tsv @@ plainto_tsquery('simple', %s)
                ORDER BY rank DESC
                LIMIT %s
            """
            params = (query_text, query_text, top_k)

        return self.db.execute_query(query, params)

    def hybrid_search(
        self,
        query_embedding: np.ndarray,
        query_text: str,
        top_k: int = 10,
        semantic_weight: float = 0.5,
        document_id: Optional[UUID] = None
    ) -> List[Dict]:
        """Perform hybrid search using the database function"""
        embedding_list = query_embedding.tolist()

        query = """
            SELECT * FROM hybrid_search(%s::vector, %s, %s, %s)
        """

        results = self.db.execute_query(
            query,
            (embedding_list, query_text, top_k, semantic_weight)
        )

        # Filter by document_id if provided
        if document_id and results:
            results = [r for r in results if str(r['document_id']) == str(document_id)]

        return results


def compute_file_hash(file_content: bytes) -> str:
    """Compute SHA256 hash of file content"""
    return hashlib.sha256(file_content).hexdigest()
