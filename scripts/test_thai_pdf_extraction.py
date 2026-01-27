"""
Test script to verify Thai PDF text extraction improvements.
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.services.pdf_processor import PDFProcessor
from loguru import logger


def test_thai_pdf_extraction():
    """Test Thai PDF text extraction with different methods."""
    
    # Initialize processor
    processor = PDFProcessor()
    
    # Test files
    test_files = [
        "test_file/pdf/ระบบซื้อ.pdf",
        "test_file/pdf/FAQ.pdf"
    ]
    
    for pdf_path in test_files:
        if not os.path.exists(pdf_path):
            logger.warning(f"Test file not found: {pdf_path}")
            continue
            
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing: {pdf_path}")
        logger.info(f"{'='*60}")
        
        try:
            # Extract using new method
            pages_data = processor.extract_text_and_images(
                pdf_path,
                extract_images=True,
                image_output_dir=f"uploads/test_extraction_{os.path.basename(pdf_path)}"
            )
            
            # Display results
            for page_data in pages_data[:3]:  # Show first 3 pages
                page_num = page_data['page_number']
                text = page_data.get('text', '')
                
                logger.info(f"\n--- Page {page_num} ---")
                logger.info(f"Has text: {page_data['has_text']}")
                logger.info(f"Has images: {page_data['has_images']}")
                logger.info(f"Text length: {len(text)}")
                
                if text:
                    # Show first 500 characters
                    preview = text[:500]
                    logger.info(f"\nText preview:\n{preview}\n...")
                    
                    # Check for garbled text
                    import re
                    thai_chars = len(re.findall(r'[ก-๙]', preview))
                    logger.info(f"Thai characters found: {thai_chars}")
                    
                    if re.search(r'[Ê][ก-๙]', preview):
                        logger.warning("⚠️  Garbled text detected!")
                    else:
                        logger.success("✓ Text appears clean")
            
            # Test full text extraction
            logger.info(f"\n--- Full Text Extraction ---")
            full_text = processor.extract_full_text(pdf_path)
            logger.info(f"Total text length: {len(full_text)}")
            logger.info(f"First 300 chars:\n{full_text[:300]}\n...")
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    logger.info("Starting Thai PDF extraction test...")
    test_thai_pdf_extraction()
    logger.info("\nTest complete!")
