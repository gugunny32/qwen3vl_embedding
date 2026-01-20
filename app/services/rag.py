from typing import List, Dict, Optional
from loguru import logger

from app.services.search import SearchService
from app.models.generator import get_generator_model
from app.config import get_settings


class RAGService:
    """
    Retrieval-Augmented Generation service.
    Combines search with generation for question answering.
    """

    def __init__(self):
        self.settings = get_settings()
        self.search_service = SearchService()
        self.generator = None

    def _ensure_generator_loaded(self):
        """Lazy load generator model"""
        if self.generator is None:
            self.generator = get_generator_model()

    def query(
        self,
        question: str,
        top_k: int = 5,
        use_reranker: bool = True,
        document_id: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 512
    ) -> Dict:
        """
        Answer a question using RAG.

        Args:
            question: User's question
            top_k: Number of contexts to retrieve
            use_reranker: Whether to use reranker
            document_id: Optional document to search within
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Dictionary with answer and metadata
        """
        self._ensure_generator_loaded()

        logger.info(f"RAG query: {question[:100]}...")

        # Retrieve relevant contexts
        search_results = self.search_service.hybrid_search(
            query=question,
            top_k=top_k,
            document_id=document_id,
            use_reranker=use_reranker
        )

        if not search_results:
            return {
                'answer': "ไม่พบข้อมูลที่เกี่ยวข้องในฐานข้อมูล กรุณาลองถามคำถามอื่นหรือเพิ่มเอกสารเข้าระบบ",
                'contexts': [],
                'source_documents': []
            }

        # Extract contexts
        contexts = [result['content'] for result in search_results]

        # Generate answer
        answer = self.generator.generate_answer(
            question=question,
            contexts=contexts,
            max_new_tokens=max_tokens,
            temperature=temperature
        )

        # Prepare source information
        sources = self._prepare_sources(search_results)

        result = {
            'answer': answer,
            'contexts': contexts,
            'source_documents': sources,
            'num_sources': len(sources)
        }

        logger.info(f"Generated answer with {len(contexts)} contexts from {len(sources)} documents")
        return result

    def query_with_conversation(
        self,
        question: str,
        conversation_history: List[Dict[str, str]],
        top_k: int = 5,
        use_reranker: bool = True,
        temperature: float = 0.7,
        max_tokens: int = 512
    ) -> Dict:
        """
        Answer a question with conversation history.

        Args:
            question: Current question
            conversation_history: List of previous Q&A pairs
            top_k: Number of contexts to retrieve
            use_reranker: Whether to use reranker
            temperature: Generation temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Dictionary with answer and metadata
        """
        self._ensure_generator_loaded()

        # For now, we'll just use the current question
        # In the future, we could reformulate the query based on conversation history
        return self.query(
            question=question,
            top_k=top_k,
            use_reranker=use_reranker,
            temperature=temperature,
            max_tokens=max_tokens
        )

    def summarize_document(
        self,
        document_id: str,
        max_tokens: int = 512
    ) -> Dict:
        """
        Generate a summary of a document.

        Args:
            document_id: Document ID to summarize
            max_tokens: Maximum tokens for summary

        Returns:
            Dictionary with summary and metadata
        """
        self._ensure_generator_loaded()

        # Get all chunks from the document
        from app.database.operations import ChunkOperations
        chunk_ops = ChunkOperations()

        chunks = chunk_ops.get_chunks_by_document(
            document_id=document_id,
            limit=100
        )

        if not chunks:
            return {
                'summary': "ไม่พบเอกสารที่ระบุ",
                'num_chunks': 0
            }

        # Combine text from chunks
        full_text = "\n\n".join([chunk['content'] for chunk in chunks])

        # Generate summary
        summary = self.generator.generate_summary(
            text=full_text[:4000],  # Limit input length
            max_new_tokens=max_tokens
        )

        return {
            'summary': summary,
            'num_chunks': len(chunks)
        }

    def _prepare_sources(self, search_results: List[Dict]) -> List[Dict]:
        """
        Prepare source document information from search results.

        Args:
            search_results: Search results from hybrid search

        Returns:
            List of source document metadata
        """
        sources = []
        seen_docs = set()

        for result in search_results:
            doc_id = str(result['document_id'])
            page_num = result.get('page_number')

            # Create unique key for document + page
            key = f"{doc_id}_{page_num}"

            if key not in seen_docs:
                sources.append({
                    'document_id': doc_id,
                    'page_number': page_num,
                    'chunk_id': str(result.get('chunk_id', result.get('id'))),
                    'score': result.get('hybrid_score', result.get('similarity', 0)),
                    'content_preview': result['content'][:200] + "..."
                })
                seen_docs.add(key)

        return sources

    def answer_with_citations(
        self,
        question: str,
        top_k: int = 5,
        use_reranker: bool = True,
        temperature: float = 0.7,
        max_tokens: int = 512
    ) -> Dict:
        """
        Answer with inline citations to sources.

        Args:
            question: User's question
            top_k: Number of contexts
            use_reranker: Use reranker
            temperature: Generation temperature
            max_tokens: Max tokens

        Returns:
            Answer with citations
        """
        result = self.query(
            question=question,
            top_k=top_k,
            use_reranker=use_reranker,
            temperature=temperature,
            max_tokens=max_tokens
        )

        # Add citation markers to answer
        # This is a simplified version - could be enhanced with better citation tracking
        answer = result['answer']
        sources = result['source_documents']

        if sources:
            citations = "\n\n**แหล่งอ้างอิง:**\n"
            for i, source in enumerate(sources, 1):
                citations += f"{i}. เอกสาร {source['document_id'][:8]}... หน้า {source['page_number']}\n"

            answer += citations

        result['answer'] = answer
        return result
