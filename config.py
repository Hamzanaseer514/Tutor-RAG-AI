"""
Tuterby AI Configuration
Memory and processing settings to prevent crashes
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # PDF Processing Settings
    MAX_FILE_SIZE_MB = 25  # Maximum PDF file size in MB
    MAX_PDF_PAGES = 100    # Maximum number of pages in PDF
    MAX_TEXT_LENGTH = 1000000  # Maximum text length (1MB)
    MAX_CHUNKS = 1000      # Maximum number of text chunks
    
    # Chunk Settings
    CHUNK_SIZE = 300       # Size of each text chunk
    CHUNK_OVERLAP = 50     # Overlap between chunks
    
    # Memory Management
    BATCH_SIZE = 5         # Process chunks in batches
    GC_FREQUENCY = 100     # Force garbage collection every N chunks
    
    # API Settings
    GROK_API_KEY = os.getenv("GROK_API_KEY")
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./tuterby.db")
    CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    @classmethod
    def validate(cls):
        """Validate configuration settings"""
        if not cls.GROK_API_KEY:
            raise ValueError("GROK_API_KEY environment variable is required")
        
        if cls.MAX_FILE_SIZE_MB <= 0:
            raise ValueError("MAX_FILE_SIZE_MB must be positive")
        
        if cls.MAX_PDF_PAGES <= 0:
            raise ValueError("MAX_PDF_PAGES must be positive")
        
        if cls.CHUNK_SIZE <= 0:
            raise ValueError("CHUNK_SIZE must be positive")
        
        return True
    
    @classmethod
    def get_memory_info(cls):
        """Get memory configuration information"""
        return {
            "max_file_size_mb": cls.MAX_FILE_SIZE_MB,
            "max_pdf_pages": cls.MAX_PDF_PAGES,
            "max_text_length": cls.MAX_TEXT_LENGTH,
            "max_chunks": cls.MAX_CHUNKS,
            "chunk_size": cls.CHUNK_SIZE,
            "chunk_overlap": cls.CHUNK_OVERLAP,
            "batch_size": cls.BATCH_SIZE
        }
