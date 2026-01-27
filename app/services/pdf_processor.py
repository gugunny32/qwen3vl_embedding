import fitz  # PyMuPDF
from PIL import Image
from typing import List, Dict, Tuple, Optional
import io
import os
import re
from loguru import logger

from app.config import get_settings


class PDFProcessor:
    """
    Process PDF documents to extract text and images.
    Enhanced with better Thai text extraction support.
    """

    def __init__(self):
        self.settings = get_settings()
        self._ocr_available = None
        self._tesseract_available = None

    def extract_text_and_images(
        self,
        pdf_path: str,
        extract_images: bool = True,
        image_output_dir: Optional[str] = None
    ) -> List[Dict]:
        """
        Extract text and images from PDF.

        Args:
            pdf_path: Path to PDF file
            extract_images: Whether to extract images
            image_output_dir: Directory to save extracted images

        Returns:
            List of page data with text and image paths
        """
        pages_data = []

        try:
            # Open PDF
            doc = fitz.open(pdf_path)
            logger.info(f"Processing PDF: {pdf_path} ({len(doc)} pages)")

            for page_num in range(len(doc)):
                page = doc[page_num]
                page_data = {
                    'page_number': page_num + 1,
                    'text': '',
                    'images': [],
                    'has_text': False,
                    'has_images': False
                }

                # Extract text with improved method
                text = self._extract_text_from_page(page, page_num)
                if text.strip():
                    page_data['text'] = text.strip()
                    page_data['has_text'] = True

                # Extract images if requested
                if extract_images:
                    images = self._extract_images_from_page(
                        page,
                        page_num,
                        image_output_dir
                    )
                    if images:
                        page_data['images'] = images
                        page_data['has_images'] = True

                # Fallback: render full page as image if no text and no images
                if extract_images and (not page_data['has_text']) and (not page_data['has_images']):
                    try:
                        # Render page at moderate resolution
                        zoom = 2.0  # ~144 DPI
                        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
                        img_bytes = pix.tobytes("png")
                        pil_image = Image.open(io.BytesIO(img_bytes))

                        image_path = None
                        if image_output_dir:
                            os.makedirs(image_output_dir, exist_ok=True)
                            image_filename = f"page_{page_num + 1}_rendered.png"
                            image_path = os.path.join(image_output_dir, image_filename)
                            pil_image.save(image_path)

                        width, height = pil_image.size
                        page_data['images'] = [
                            {
                                'image_index': 0,
                                'image_path': image_path,
                                'width': width,
                                'height': height,
                                'format': 'png',
                                'pil_image': pil_image,
                                'rendered_page': True
                            }
                        ]
                        page_data['has_images'] = True
                        logger.info(f"Rendered page {page_num + 1} as image for OCR/caption fallback")
                    except Exception as e:
                        logger.warning(f"Failed to render page {page_num + 1} as image: {e}")

                pages_data.append(page_data)

            doc.close()
            logger.info(f"Extracted {len(pages_data)} pages from PDF")

        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {e}")
            raise

        return pages_data

    def _extract_images_from_page(
        self,
        page: fitz.Page,
        page_num: int,
        output_dir: Optional[str] = None
    ) -> List[Dict]:
        """
        Extract images from a PDF page.

        Args:
            page: PyMuPDF page object
            page_num: Page number
            output_dir: Directory to save images

        Returns:
            List of image metadata
        """
        images_data = []

        # Get images from page (full=True includes inline images)
        image_list = page.get_images(full=True)

        for img_idx, img in enumerate(image_list):
            try:
                xref = img[0]
                base_image = page.parent.extract_image(xref)

                # Get image data
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                # Convert to PIL Image
                pil_image = Image.open(io.BytesIO(image_bytes))

                # Filter out tiny images (likely icons) with lower threshold
                width, height = pil_image.size
                if width < 10 or height < 10:
                    continue

                # Save image if output directory provided
                image_path = None
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    image_filename = f"page_{page_num + 1}_img_{img_idx + 1}.{image_ext}"
                    image_path = os.path.join(output_dir, image_filename)
                    pil_image.save(image_path)

                images_data.append({
                    'image_index': img_idx,
                    'image_path': image_path,
                    'width': width,
                    'height': height,
                    'format': image_ext,
                    'pil_image': pil_image
                })

            except Exception as e:
                logger.warning(f"Failed to extract image {img_idx} from page {page_num}: {e}")
                continue

        if images_data:
            logger.info(f"Extracted {len(images_data)} images from page {page_num + 1}")
        return images_data

    def get_pdf_metadata(self, pdf_path: str) -> Dict:
        """
        Extract metadata from PDF.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Dictionary of metadata
        """
        try:
            doc = fitz.open(pdf_path)
            metadata = doc.metadata
            num_pages = len(doc)
            doc.close()

            return {
                'title': metadata.get('title', ''),
                'author': metadata.get('author', ''),
                'subject': metadata.get('subject', ''),
                'creator': metadata.get('creator', ''),
                'producer': metadata.get('producer', ''),
                'num_pages': num_pages
            }
        except Exception as e:
            logger.error(f"Error extracting metadata from {pdf_path}: {e}")
            return {}

    def extract_text_by_page(self, pdf_path: str) -> List[str]:
        """
        Extract text from each page separately.

        Args:
            pdf_path: Path to PDF file

        Returns:
            List of page texts
        """
        page_texts = []

        try:
            doc = fitz.open(pdf_path)

            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                page_texts.append(text.strip())

            doc.close()

        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            raise

        return page_texts

    def extract_full_text(self, pdf_path: str) -> str:
        """
        Extract all text from PDF as single string.

        Args:
            pdf_path: Path to PDF file

        Returns:
            Full text content
        """
        try:
            doc = fitz.open(pdf_path)
            full_text = ""

            for page_num, page in enumerate(doc):
                page_text = self._extract_text_from_page(page, page_num)
                full_text += page_text + "\n\n"

            doc.close()
            return full_text.strip()

        except Exception as e:
            logger.error(f"Error extracting full text from {pdf_path}: {e}")
            raise

    def _extract_text_from_page(
        self,
        page: fitz.Page,
        page_num: int,
        use_ocr_fallback: bool = True
    ) -> str:
        """
        Extract text from a page with multiple strategies for Thai text support.

        Tries multiple extraction methods:
        1. Standard text extraction with different flags
        2. Block-based extraction
        3. OCR fallback for garbled text (especially Thai)

        Args:
            page: PyMuPDF page object
            page_num: Page number
            use_ocr_fallback: Whether to use OCR when text is garbled

        Returns:
            Extracted text
        """
        # Strategy 1: Try standard extraction with flags
        try:
            flags = fitz.TEXT_PRESERVE_LIGATURES | fitz.TEXT_PRESERVE_WHITESPACE
            text = page.get_text("text", flags=flags)
            
            # Check if text is garbled (contains replacement characters or encoding issues)
            if text and self._is_text_garbled(text):
                logger.debug(f"Page {page_num + 1}: Detected garbled text, trying alternative methods")
                
                # Strategy 2: Try block-based extraction
                text_blocks = page.get_text("blocks")
                block_text = "\n".join([block[4] for block in text_blocks if len(block) > 4 and block[4].strip()])
                
                if block_text and not self._is_text_garbled(block_text):
                    logger.debug(f"Page {page_num + 1}: Block extraction successful")
                    return block_text
                
                # Strategy 3: OCR fallback for garbled Thai text
                if use_ocr_fallback:
                    logger.info(f"Page {page_num + 1}: Using OCR fallback for Thai text extraction")
                    ocr_text = self._ocr_page(page, page_num)
                    if ocr_text:
                        return ocr_text
                    
            return text
            
        except Exception as e:
            logger.warning(f"Error extracting text from page {page_num}: {e}")
            return ""

    def _is_text_garbled(self, text: str) -> bool:
        """
        Check if extracted text contains garbled characters.
        Common issue with Thai PDFs using embedded fonts.

        Args:
            text: Text to check

        Returns:
            True if text appears garbled
        """
        if not text:
            return False
        
        # Check for common garbled character patterns
        garbled_patterns = [
            r'[\uFFFD]',  # Replacement character
            r'[Ê][ก-๙]',  # Common Thai garbling pattern (Ê followed by Thai)
            r'[\u0080-\u009F]',  # Control characters
        ]
        
        for pattern in garbled_patterns:
            if re.search(pattern, text):
                return True
        
        # Check ratio of Thai characters vs total
        # If we have Thai-like content but it's mixed with weird chars, it's likely garbled
        thai_chars = len(re.findall(r'[ก-๙]', text))
        if thai_chars > 0:
            # Look for suspicious adjacent character combinations
            if re.search(r'[ก-๙][A-Z][ก-๙]', text) or re.search(r'[Ê][ก-๙]', text):
                return True
                
        return False

    def _ocr_page(self, page: fitz.Page, page_num: int) -> str:
        """
        Perform OCR on a page to extract text.
        Uses pytesseract with Thai language support.

        Args:
            page: PyMuPDF page object
            page_num: Page number

        Returns:
            OCR extracted text
        """
        try:
            # Check if pytesseract is available (lazy load)
            if self._tesseract_available is None:
                try:
                    import pytesseract
                    self._tesseract_available = True
                except ImportError:
                    logger.warning("pytesseract not available, OCR fallback disabled")
                    self._tesseract_available = False
            
            if not self._tesseract_available:
                return ""
            
            import pytesseract
            
            # Render page to image at high resolution for better OCR
            zoom = 2.5  # ~180 DPI
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)
            img_bytes = pix.tobytes("png")
            pil_image = Image.open(io.BytesIO(img_bytes))
            
            # Perform OCR with Thai + English
            # You can adjust languages as needed
            try:
                text = pytesseract.image_to_string(
                    pil_image,
                    lang='tha+eng',  # Thai + English
                    config='--psm 6'  # Assume uniform block of text
                )
            except Exception as ocr_error:
                # Fallback to Thai only
                logger.debug(f"OCR with tha+eng failed, trying tha only: {ocr_error}")
                text = pytesseract.image_to_string(
                    pil_image,
                    lang='tha',
                    config='--psm 6'
                )
            
            if text.strip():
                logger.info(f"Page {page_num + 1}: OCR extracted {len(text)} characters")
                return text.strip()
                
        except Exception as e:
            logger.warning(f"OCR failed for page {page_num + 1}: {e}")
        
        return ""

    def _check_ocr_available(self) -> bool:
        """Check if OCR dependencies are available."""
        if self._ocr_available is not None:
            return self._ocr_available
            
        try:
            import pytesseract
            # Try to get tesseract version
            pytesseract.get_tesseract_version()
            self._ocr_available = True
            logger.info("Tesseract OCR is available")
        except Exception:
            self._ocr_available = False
            logger.warning("Tesseract OCR not available - install pytesseract and tesseract-ocr for Thai text support")
        
        return self._ocr_available
