import PyPDF2
import pdfplumber
from pdf2image import convert_from_path
import pytesseract
from PIL import Image, ImageEnhance
from typing import List
import logging
import io
import gc
import re
import fitz  # PyMuPDF for better text extraction

logger = logging.getLogger(__name__)

class PDFProcessor:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_total_chunks = 1000  # Limit total chunks to prevent memory issues
    
    def process_pdf(self, file_path: str, password: str = None) -> List[str]:
        logger.info(f"Processing PDF: {file_path}")
        
        # Extract text
        text = self._extract_text_from_pdf(file_path, password)
        logger.info(f"Extracted text length: {len(text)} characters")
        
        if not text.strip():
            logger.error("No text could be extracted from PDF!")
            return []
        
        # Clean text quickly
        text = self._clean_text(text)
        logger.info(f"Cleaned text length: {len(text)} characters")
        
        # Create chunks quickly
        logger.info("Starting text chunking...")
        chunks = self._chunk_text(text)
        logger.info(f"Created {len(chunks)} chunks")
        
        # Limit chunks if too many
        if len(chunks) > 50:  # Reduced from 1000 to 50 for speed
            logger.warning(f"Limiting chunks from {len(chunks)} to 50 for faster processing")
            chunks = chunks[:50]
        
        # Log first few chunks
        for i, chunk in enumerate(chunks[:3]):
            logger.info(f"Chunk {i+1}: {chunk[:100]}...")
        
        logger.info(f"PDF processing completed. Returning {len(chunks)} chunks.")
        return chunks
    
    def _extract_text_from_pdf(self, file_path: str, password: str = None) -> str:
        """Extract text from PDF using multiple methods"""
        text = ""
        
        # Method 1: Try PyMuPDF (fitz) first - most reliable
        try:
            logger.info("Trying PyMuPDF text extraction...")
            import fitz
            doc = fitz.open(file_path)
            if password:
                doc.authenticate(password)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                if page_text.strip():
                    text += page_text + "\n"
            
            doc.close()
            if text.strip():
                logger.info("PyMuPDF text extraction successful")
                return text.strip()
            else:
                logger.warning("PyMuPDF extracted empty text")
        except Exception as e:
            logger.warning(f"PyMuPDF failed: {str(e)}")
        
        # Method 2: Try PyPDF2
        try:
            logger.info("Trying PyPDF2 text extraction...")
            import PyPDF2
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                if password:
                    reader.decrypt(password)
                
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            
            if text.strip():
                logger.info("PyPDF2 text extraction successful")
                return text.strip()
            else:
                logger.warning("PyPDF2 extracted empty text")
        except Exception as e:
            logger.warning(f"PyPDF2 failed: {str(e)}")
        
        # Method 3: Try pdfplumber
        try:
            logger.info("Trying pdfplumber text extraction...")
            import pdfplumber
            with pdfplumber.open(file_path, password=password) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            
            if text.strip():
                logger.info("pdfplumber text extraction successful")
                return text.strip()
            else:
                logger.warning("pdfplumber extracted empty text")
        except Exception as e:
            logger.warning(f"pdfplumber failed: {str(e)}")
        
        # Method 4: OCR (DISABLED - requires Poppler)
        # try:
        #     logger.info("Trying OCR text extraction...")
        #     from pdf2image import convert_from_path
        #     import pytesseract
        #     
        #     images = convert_from_path(file_path)
        #     ocr_text = ""
        #     
        #     for i, image in enumerate(images):
        #         page_text = pytesseract.image_to_string(image)
        #         if page_text.strip():
        #             ocr_text += f"Page {i+1}:\n{page_text}\n\n"
        #     
        #     if ocr_text.strip():
        #         logger.info("OCR text extraction successful")
        #         return ocr_text.strip()
        #     else:
        #         logger.warning("OCR extracted empty text")
        # except Exception as e:
        #     logger.warning(f"OCR failed: {str(e)}")
        
        # If all methods failed
        if not text.strip():
            raise ValueError("All text extraction methods failed. The PDF might be scanned, protected, or corrupted.")
        
        return text.strip()
    
    def _clean_text(self, text: str) -> str:
        """Clean and preprocess extracted text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep Urdu and English
        text = re.sub(r'[^\w\s\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF\.\,\!\?\:\;\(\)\[\]\{\}\-\+\=\*\/\\\@\#\$\%\&\|\<\>\'\"\n]', '', text)
        
        # Remove page numbers and headers
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
        
        # Remove excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    def _chunk_text(self, text: str) -> List[str]:
        """Create simple, fast text chunks"""
        logger.info("Starting simple text chunking...")
        
        chunks = []
        text_length = len(text)
        logger.info(f"Text length: {text_length} characters")
        
        # Simple chunking: just split by character count
        chunk_size = self.chunk_size
        overlap = self.chunk_overlap
        
        logger.info(f"Using chunk size: {chunk_size}, overlap: {overlap}")
        
        # Create chunks quickly without complex logic
        start = 0
        chunk_count = 0
        
        while start < text_length and chunk_count < 100:  # Limit to 100 chunks max
            end = start + chunk_size
            
            # Get the chunk
            chunk = text[start:end].strip()
            
            # Only add if chunk is meaningful
            if len(chunk) > 20:  # Minimum chunk size
                chunks.append(chunk)
                chunk_count += 1
                
                # Log progress
                if chunk_count % 10 == 0:
                    logger.info(f"Created {chunk_count} chunks...")
            
            # Move to next chunk with overlap
            start = end - overlap
            if start >= text_length:
                break
        
        logger.info(f"Simple chunking completed. Created {len(chunks)} chunks.")
        return chunks