from typing import List, Dict, Optional
import re
from loguru import logger

from app.config import get_settings


class TextChunker:
    """
    Intelligent text chunking with overlap for better context preservation.
    """

    def __init__(self):
        self.settings = get_settings()
        self.max_chunk_size = self.settings.max_chunk_size
        self.chunk_overlap = self.settings.chunk_overlap

    def chunk_text(
        self,
        text: str,
        max_size: Optional[int] = None,
        overlap: Optional[int] = None
    ) -> List[str]:
        """
        Split text into chunks with overlap.

        Args:
            text: Text to chunk
            max_size: Maximum chunk size (default from config)
            overlap: Overlap size (default from config)

        Returns:
            List of text chunks
        """
        max_size = max_size or self.max_chunk_size
        overlap = overlap or self.chunk_overlap

        # Clean text
        text = self._clean_text(text)

        if not text:
            return []

        # Try sentence-based chunking first
        chunks = self._chunk_by_sentences(text, max_size, overlap)

        # If that fails, fall back to simple chunking
        if not chunks:
            chunks = self._chunk_by_characters(text, max_size, overlap)

        logger.debug(f"Created {len(chunks)} chunks from text of length {len(text)}")
        return chunks

    def chunk_pages(
        self,
        pages_data: List[Dict],
        include_page_context: bool = True
    ) -> List[Dict]:
        """
        Chunk text from multiple pages with metadata.

        Args:
            pages_data: List of page data dicts
            include_page_context: Whether to include page number in chunks

        Returns:
            List of chunk dicts with metadata
        """
        all_chunks = []
        chunk_index = 0

        for page_data in pages_data:
            page_num = page_data['page_number']
            text = page_data.get('text', '')

            if not text:
                # If no text but has images, create a placeholder chunk
                if page_data.get('has_images'):
                    for img_data in page_data.get('images', []):
                        all_chunks.append({
                            'chunk_index': chunk_index,
                            'content': f"[Image on page {page_num}]",
                            'page_number': page_num,
                            'has_image': True,
                            'image_path': img_data.get('image_path'),
                            'pil_image': img_data.get('pil_image'),
                            'metadata': {
                                'page': page_num,
                                'image_index': img_data.get('image_index'),
                                'image_dimensions': f"{img_data.get('width')}x{img_data.get('height')}"
                            }
                        })
                        chunk_index += 1
                continue

            # Add page context if requested
            if include_page_context:
                text = f"[หน้า {page_num}]\n{text}"

            # Chunk the text
            text_chunks = self.chunk_text(text)

            # Create chunk metadata
            for chunk_text in text_chunks:
                chunk_dict = {
                    'chunk_index': chunk_index,
                    'content': chunk_text,
                    'page_number': page_num,
                    'has_image': False,
                    'image_path': None,
                    'pil_image': None,
                    'metadata': {
                        'page': page_num,
                        'has_images': page_data.get('has_images', False)
                    }
                }

                all_chunks.append(chunk_dict)
                chunk_index += 1

            # Add image chunks if present
            if page_data.get('has_images'):
                for img_data in page_data.get('images', []):
                    # Create a multimodal chunk with text context and image
                    image_context = f"[หน้า {page_num}] รูปภาพที่ {img_data.get('image_index', 0) + 1}\n"

                    # Try to get surrounding text as context
                    if text_chunks:
                        # Use the last text chunk as context
                        image_context += text_chunks[-1][:200] + "..."

                    all_chunks.append({
                        'chunk_index': chunk_index,
                        'content': image_context,
                        'page_number': page_num,
                        'has_image': True,
                        'image_path': img_data.get('image_path'),
                        'pil_image': img_data.get('pil_image'),
                        'metadata': {
                            'page': page_num,
                            'image_index': img_data.get('image_index'),
                            'image_dimensions': f"{img_data.get('width')}x{img_data.get('height')}"
                        }
                    })
                    chunk_index += 1

        logger.info(f"Created {len(all_chunks)} chunks from {len(pages_data)} pages")
        return all_chunks

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove leading/trailing whitespace
        text = text.strip()
        return text

    def _chunk_by_sentences(
        self,
        text: str,
        max_size: int,
        overlap: int
    ) -> List[str]:
        """
        Chunk text by sentences to preserve semantic boundaries.
        """
        # Split into sentences (handles Thai and English)
        sentences = re.split(r'([.!?。！？\n]+)', text)

        # Recombine sentences with their punctuation
        sentences = [sentences[i] + (sentences[i+1] if i+1 < len(sentences) else '')
                    for i in range(0, len(sentences), 2)]

        chunks = []
        current_chunk = ""
        overlap_buffer = []

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # If adding this sentence exceeds max_size
            if len(current_chunk) + len(sentence) > max_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())

                    # Keep last few sentences for overlap
                    overlap_text = " ".join(overlap_buffer[-2:])
                    current_chunk = overlap_text + " " + sentence if overlap_text else sentence
                    overlap_buffer = [sentence]
                else:
                    # Single sentence is too long, split it
                    chunks.append(sentence[:max_size])
                    current_chunk = sentence[max_size:]
                    overlap_buffer = []
            else:
                current_chunk += " " + sentence if current_chunk else sentence
                overlap_buffer.append(sentence)

        # Add remaining chunk
        if current_chunk:
            chunks.append(current_chunk.strip())

        return chunks

    def _chunk_by_characters(
        self,
        text: str,
        max_size: int,
        overlap: int
    ) -> List[str]:
        """
        Simple character-based chunking with overlap.
        """
        chunks = []
        start = 0

        while start < len(text):
            end = start + max_size
            chunk = text[start:end]

            # Try to break at word boundary
            if end < len(text):
                last_space = chunk.rfind(' ')
                if last_space > max_size * 0.8:  # If space is in last 20%
                    end = start + last_space

            chunks.append(text[start:end].strip())
            start = end - overlap

        return chunks

    def merge_short_chunks(
        self,
        chunks: List[str],
        min_size: int = 100
    ) -> List[str]:
        """
        Merge chunks that are too short with adjacent chunks.
        """
        if not chunks:
            return []

        merged = []
        current = chunks[0]

        for next_chunk in chunks[1:]:
            if len(current) < min_size:
                current += " " + next_chunk
            else:
                merged.append(current)
                current = next_chunk

        # Add last chunk
        if current:
            if merged and len(current) < min_size:
                merged[-1] += " " + current
            else:
                merged.append(current)

        return merged
