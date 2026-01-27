#!/bin/bash
# Quick test script for Thai PDF extraction
# Run this inside the Docker container

echo "==================================="
echo "Thai PDF Extraction Test"
echo "==================================="
echo ""

# Test 1: Check Tesseract installation
echo "1. Checking Tesseract installation..."
tesseract --version
echo ""

echo "2. Available OCR languages:"
tesseract --list-langs
echo ""

# Test 2: Quick Python test
echo "3. Testing Python PDF extraction..."
python3 -c "
from app.services.pdf_processor import PDFProcessor
from loguru import logger
import sys

processor = PDFProcessor()
pdf_path = 'test_file/pdf/ระบบซื้อ.pdf'

try:
    # Test extraction
    pages = processor.extract_text_and_images(pdf_path, extract_images=False)
    
    if pages and pages[0].get('text'):
        text = pages[0]['text']
        print(f'✓ Successfully extracted text from first page')
        print(f'  Length: {len(text)} characters')
        print(f'  Preview: {text[:200]}...')
        
        # Check for Thai characters
        import re
        thai_chars = len(re.findall(r'[ก-๙]', text))
        print(f'  Thai characters: {thai_chars}')
        
        # Check for garbled text
        if re.search(r'[Ê][ก-๙]', text):
            print('  ⚠️  Warning: Garbled text detected!')
            sys.exit(1)
        else:
            print('  ✓ Text appears clean!')
            sys.exit(0)
    else:
        print('✗ No text extracted')
        sys.exit(1)
        
except Exception as e:
    print(f'✗ Error: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "==================================="
    echo "✓ All tests passed!"
    echo "==================================="
else
    echo ""
    echo "==================================="
    echo "✗ Tests failed!"
    echo "==================================="
    exit 1
fi
