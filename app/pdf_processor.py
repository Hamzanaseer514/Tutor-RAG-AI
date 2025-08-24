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
        text = ""
        
        # Try PyMuPDF first (best text extraction)
        try:
            logger.info("Trying PyMuPDF for text extraction...")
            doc = fitz.open(file_path)
            if doc.needs_pass:
                if password:
                    doc.authenticate(password)
                else:
                    raise ValueError("PDF is password-protected. Please provide a password.")
            
            for page_num in range(len(doc)):
                if page_num % 10 == 0:
                    logger.info(f"Processing page {page_num+1}/{len(doc)} with PyMuPDF")
                page = doc.load_page(page_num)
                page_text = page.get_text()
                if page_text:
                    text += page_text + "\n"
                
                # Check if text is getting too long and truncate if necessary
                if len(text) > 1000000:  # 1MB limit
                    logger.warning("Text too long, truncating to prevent memory issues")
                    text = text[:1000000]
                    break
            
            doc.close()
            
            if len(text.strip()) > 100:
                logger.info("PyMuPDF extraction successful")
                return text
                
        except Exception as e:
            logger.warning(f"PyMuPDF failed: {str(e)}")
        
        # Try pdfplumber as fallback
        try:
            logger.info("Trying pdfplumber for text extraction...")
            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    if i % 10 == 0:
                        logger.info(f"Processing page {i+1}/{len(pdf.pages)} with pdfplumber")
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                    
                    # Check text length limit
                    if len(text) > 1000000:
                        logger.warning("Text too long, truncating to prevent memory issues")
                        text = text[:1000000]
                        break
            
            if len(text.strip()) > 100:
                logger.info("pdfplumber extraction successful")
                return text
                
        except Exception as e:
            logger.warning(f"pdfplumber failed: {str(e)}")
        
        # Try PyPDF2 as fallback
        try:
            logger.info("Trying PyPDF2 for text extraction...")
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                if pdf_reader.is_encrypted:
                    try:
                        pdf_reader.decrypt(password or "")
                    except Exception as e:
                        logger.error(f"PDF decryption failed: {str(e)}")
                        raise ValueError("PDF is password-protected and provided password is incorrect.")
                
                total_pages = len(pdf_reader.pages)
                for i, page in enumerate(pdf_reader.pages):
                    if i % 10 == 0:
                        logger.info(f"Processing page {i+1}/{total_pages} with PyPDF2")
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                    
                    # Check text length limit
                    if len(text) > 1000000:
                        logger.warning("Text too long, truncating to prevent memory issues")
                        text = text[:1000000]
                        break
            
            if len(text.strip()) > 100:
                logger.info("PyPDF2 extraction successful")
                return text
                
        except Exception as e:
            logger.warning(f"PyPDF2 failed: {str(e)}")
        
        # Try OCR as last resort
        if len(text.strip()) < 100:
            logger.info("Text extraction methods failed, trying Tesseract OCR...")
            try:
                images = convert_from_path(file_path)
                for i, image in enumerate(images):
                    if i % 10 == 0:
                        logger.info(f"Processing page {i+1}/{len(images)} with OCR")
                    # Enhance image for better OCR
                    image = ImageEnhance.Contrast(image).enhance(2.0)
                    image = ImageEnhance.Sharpness(image).enhance(1.5)
                    page_text = pytesseract.image_to_string(image, lang='eng+urd')
                    if page_text:
                        text += page_text + "\n"
                    
                    # Check text length limit
                    if len(text) > 1000000:
                        logger.warning("Text too long, truncating to prevent memory issues")
                        text = text[:1000000]
                        break
                        
                logger.info("OCR processing completed")
                
            except Exception as e:
                logger.error(f"OCR processing failed: {str(e)}")
                raise ValueError(f"All text extraction methods failed. OCR error: {str(e)}")
        
        if len(text.strip()) < 100:
            raise ValueError("Could not extract sufficient text from PDF. The document might be corrupted or contain only images.")
        
        return text
    
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