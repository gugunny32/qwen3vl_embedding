from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from typing import Optional, List
from pydantic import BaseModel
import os
import shutil
from uuid import UUID
from loguru import logger

from app.database.operations import DocumentOperations, ChunkOperations, compute_file_hash
from app.services.pdf_processor import PDFProcessor
from app.services.chunker import TextChunker
from app.models.embedding import get_embedding_model
from app.config import get_settings

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])
settings = get_settings()


class DocumentResponse(BaseModel):
    id: str
    filename: str
    title: Optional[str]
    total_chunks: int
    file_size: Optional[int]
    created_at: str


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    message: str
    total_chunks: int


class ChunkResponse(BaseModel):
    id: str
    chunk_index: int
    content: str
    page_number: Optional[int]
    has_image: bool


def process_pdf_document(
    file_path: str,
    filename: str,
    file_content: bytes,
    title: Optional[str] = None
):
    """Background task to process PDF document"""
    try:
        logger.info(f"Processing document: {filename}")

        # Initialize services
        doc_ops = DocumentOperations()
        chunk_ops = ChunkOperations()
        pdf_processor = PDFProcessor()
        chunker = TextChunker()
        embedding_model = get_embedding_model()

        # Check if document already exists by hash
        content_hash = compute_file_hash(file_content)
        existing_doc_id = doc_ops.document_exists_by_hash(content_hash)

        if existing_doc_id:
            logger.info(f"Document with same hash already exists: {existing_doc_id}")
            # Clean up uploaded file
            if os.path.exists(file_path):
                os.remove(file_path)
            return

        # Create document record
        doc_id = doc_ops.create_document(
            filename=filename,
            title=title or filename,
            file_size=len(file_content),
            content_hash=content_hash
        )

        # Extract images directory
        image_dir = os.path.join(settings.upload_dir, f"images_{doc_id}")

        # Extract text and images from PDF
        pages_data = pdf_processor.extract_text_and_images(
            file_path,
            extract_images=True,
            image_output_dir=image_dir
        )

        # Chunk the pages
        chunks_data = chunker.chunk_pages(pages_data)

        # Generate embeddings and store chunks
        for chunk_data in chunks_data:
            # Generate embedding
            if chunk_data['has_image'] and chunk_data['pil_image']:
                # Multimodal embedding
                embedding = embedding_model.encode_multimodal(
                    text=chunk_data['content'],
                    image=chunk_data['pil_image']
                )
            else:
                # Text-only embedding
                embedding = embedding_model.encode_text(chunk_data['content'])[0]

            # Store chunk
            chunk_ops.create_chunk(
                document_id=doc_id,
                chunk_index=chunk_data['chunk_index'],
                content=chunk_data['content'],
                embedding=embedding,
                metadata=chunk_data['metadata'],
                has_image=chunk_data['has_image'],
                image_path=chunk_data['image_path'],
                page_number=chunk_data['page_number']
            )

        # Update document with chunk count
        doc_ops.update_document_chunks_count(doc_id, len(chunks_data))

        logger.info(f"Successfully processed document {doc_id} with {len(chunks_data)} chunks")

        # Clean up original PDF file
        if os.path.exists(file_path):
            os.remove(file_path)

    except Exception as e:
        logger.error(f"Error processing document {filename}: {e}")
        # Clean up on error
        if os.path.exists(file_path):
            os.remove(file_path)
        raise


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: Optional[str] = None
):
    """
    Upload a PDF document for processing.
    """
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Create upload directory if not exists
    os.makedirs(settings.upload_dir, exist_ok=True)

    # Save uploaded file
    file_path = os.path.join(settings.upload_dir, file.filename)

    try:
        # Read file content
        file_content = await file.read()

        # Check file size
        if len(file_content) > settings.max_file_size:
            raise HTTPException(
                status_code=400,
                detail=f"File size exceeds maximum allowed size ({settings.max_file_size} bytes)"
            )

        # Save to disk
        with open(file_path, "wb") as f:
            f.write(file_content)

        # Process in background
        background_tasks.add_task(
            process_pdf_document,
            file_path,
            file.filename,
            file_content,
            title
        )

        return DocumentUploadResponse(
            document_id="processing",
            filename=file.filename,
            status="processing",
            message="Document is being processed in the background",
            total_chunks=0
        )

    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        # Clean up on error
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", response_model=List[DocumentResponse])
async def list_documents(limit: int = 100, offset: int = 0):
    """
    List all documents.
    """
    try:
        doc_ops = DocumentOperations()
        documents = doc_ops.get_all_documents(limit=limit, offset=offset)

        return [
            DocumentResponse(
                id=str(doc['id']),
                filename=doc['filename'],
                title=doc.get('title'),
                total_chunks=doc.get('total_chunks', 0),
                file_size=doc.get('file_size'),
                created_at=str(doc['created_at'])
            )
            for doc in documents
        ]
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(document_id: str):
    """
    Get a specific document by ID.
    """
    try:
        doc_ops = DocumentOperations()
        document = doc_ops.get_document(UUID(document_id))

        if not document:
            raise HTTPException(status_code=404, detail="Document not found")

        return DocumentResponse(
            id=str(document['id']),
            filename=document['filename'],
            title=document.get('title'),
            total_chunks=document.get('total_chunks', 0),
            file_size=document.get('file_size'),
            created_at=str(document['created_at'])
        )
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID format")
    except Exception as e:
        logger.error(f"Error getting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{document_id}/chunks", response_model=List[ChunkResponse])
async def get_document_chunks(
    document_id: str,
    limit: int = 100,
    offset: int = 0
):
    """
    Get chunks for a specific document.
    """
    try:
        chunk_ops = ChunkOperations()
        chunks = chunk_ops.get_chunks_by_document(
            document_id=UUID(document_id),
            limit=limit,
            offset=offset
        )

        return [
            ChunkResponse(
                id=str(chunk['id']),
                chunk_index=chunk['chunk_index'],
                content=chunk['content'],
                page_number=chunk.get('page_number'),
                has_image=chunk.get('has_image', False)
            )
            for chunk in chunks
        ]
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID format")
    except Exception as e:
        logger.error(f"Error getting chunks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{document_id}")
async def delete_document(document_id: str):
    """
    Delete a document and all its chunks.
    """
    try:
        doc_ops = DocumentOperations()
        doc_ops.delete_document(UUID(document_id))

        # Clean up image directory
        image_dir = os.path.join(settings.upload_dir, f"images_{document_id}")
        if os.path.exists(image_dir):
            shutil.rmtree(image_dir)

        return {"message": "Document deleted successfully", "document_id": document_id}
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid document ID format")
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=str(e))
