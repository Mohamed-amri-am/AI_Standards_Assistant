"""
Configuration settings for AI Standards Assistant

This module contains all configuration settings that can be customized
for different environments and use cases.
"""

import os
from pathlib import Path
from typing import Dict, Any

# Base paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
STANDARDS_DIR = DATA_DIR / "standards"
INDEXING_DIR = PROJECT_ROOT / "indexing"
OCR_DIR = PROJECT_ROOT / "ocr"
API_DIR = PROJECT_ROOT / "api"
UTILS_DIR = PROJECT_ROOT / "utils"

# Ensure directories exist
STANDARDS_DIR.mkdir(parents=True, exist_ok=True)
INDEXING_DIR.mkdir(parents=True, exist_ok=True)

# OCR Configuration
OCR_CONFIG = {
    "tesseract_cmd": os.getenv("TESSERACT_CMD", "tesseract"),
    "ocr_language": "eng",
    "min_text_length": 50,  # Minimum characters to consider as text
    "dpi": 300,  # DPI for OCR image conversion
}

# Text Extraction Configuration
TEXT_EXTRACTION_CONFIG = {
    "min_section_length": 200,  # Minimum characters per section
    "max_section_length": 2000,  # Maximum characters per section
    "batch_size": 32,  # Batch size for processing
}

# Embedding Configuration
EMBEDDING_CONFIG = {
    "model_name": "all-MiniLM-L6-v2",
    "device": "auto",  # "auto", "cpu", "cuda"
    "batch_size": 32,  # Batch size for embedding generation
    "normalize_embeddings": True,
}

# FAISS Index Configuration
FAISS_CONFIG = {
    "index_type": "IndexFlatIP",  # Inner product for cosine similarity
    "similarity_metric": "cosine",
    "normalize_vectors": True,
}

# API Configuration
API_CONFIG = {
    "host": os.getenv("API_HOST", "0.0.0.0"),
    "port": int(os.getenv("API_PORT", "8000")),
    "reload": os.getenv("API_RELOAD", "true").lower() == "true",
    "log_level": os.getenv("API_LOG_LEVEL", "info"),
    "cors_origins": ["*"],  # In production, specify actual origins
    "max_results": 20,  # Maximum number of search results
    "default_results": 5,  # Default number of search results
}

# Search Configuration
SEARCH_CONFIG = {
    "default_top_k": 5,
    "max_top_k": 20,
    "min_similarity_score": 0.1,  # Minimum similarity score to return
    "include_preview_default": False,
    "preview_dpi": 150,
    "preview_format": "PNG",
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": os.getenv("LOG_LEVEL", "INFO"),
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": None,  # Set to a file path to log to file
}

# File Paths
FILE_PATHS = {
    "extracted_text": INDEXING_DIR / "extracted_text.json",
    "faiss_index": INDEXING_DIR / "faiss_index.bin",
    "standards_data": INDEXING_DIR / "standards_data.json",
    "index_info": INDEXING_DIR / "index_info.json",
}

# PDF Processing Configuration
PDF_CONFIG = {
    "supported_formats": [".pdf"],
    "max_file_size_mb": 100,  # Maximum PDF file size
    "extraction_methods": ["direct", "ocr"],  # Order of extraction methods to try
    "image_formats": ["PNG", "JPEG"],
    "default_image_dpi": 150,
    "highlight_color": (1.0, 1.0, 0.0),  # Yellow highlight (RGB 0-1)
}

# Performance Configuration
PERFORMANCE_CONFIG = {
    "enable_caching": True,
    "cache_size_mb": 100,
    "parallel_processing": True,
    "max_workers": None,  # None for auto-detection
    "chunk_size": 1000,  # For processing large files
}

# Development Configuration
DEV_CONFIG = {
    "debug_mode": os.getenv("DEBUG", "false").lower() == "true",
    "test_data_dir": PROJECT_ROOT / "test_data",
    "sample_queries": [
        "noise level in workshop exceeds the limit",
        "safety requirements for electrical equipment",
        "fire prevention measures in industrial facilities",
        "maximum temperature limits for machinery",
        "personal protective equipment requirements",
        "electrical safety standards",
        "environmental regulations",
        "quality control procedures",
        "maintenance requirements",
        "emergency procedures"
    ]
}

# Model-specific configurations
MODEL_CONFIGS = {
    "all-MiniLM-L6-v2": {
        "dimension": 384,
        "max_sequence_length": 512,
        "description": "Fast and efficient model for general use"
    },
    "all-mpnet-base-v2": {
        "dimension": 768,
        "max_sequence_length": 512,
        "description": "Higher quality but slower model"
    },
    "paraphrase-multilingual-MiniLM-L12-v2": {
        "dimension": 384,
        "max_sequence_length": 512,
        "description": "Multilingual model for non-English documents"
    }
}

# Section detection patterns
SECTION_PATTERNS = [
    r'^\s*(?:Chapter|CHAPTER)\s+(\d+[\.\-\s]*[A-Za-z\s]+)',
    r'^\s*(\d+\.\d*)\s+([A-Z][A-Za-z\s]+)',
    r'^\s*([A-Z][A-Z\s]{3,})',
    r'^\s*(?:Section|SECTION)\s+(\d+[\.\-\s]*[A-Za-z\s]+)',
    r'^\s*(\d+\.\d+\.\d*)\s+([A-Z][A-Za-z\s]+)',
    r'^\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*$'
]

# Error messages
ERROR_MESSAGES = {
    "no_pdfs_found": "No PDF files found in standards folder. Please add PDF files to data/standards/",
    "index_not_found": "Search index not found. Please run the indexing process first.",
    "extraction_failed": "Text extraction failed. Please check your PDF files and try again.",
    "indexing_failed": "Index building failed. Please check the extracted text data.",
    "api_not_ready": "API server is not ready. Please wait for initialization to complete.",
    "invalid_query": "Invalid search query. Please provide a non-empty query string.",
    "document_not_found": "Document not found. Please check the document name.",
    "page_not_found": "Page not found. Please check the page number.",
    "tesseract_not_found": "Tesseract OCR not found. Please install Tesseract OCR.",
    "cuda_not_available": "CUDA not available. Using CPU for processing.",
}

# Success messages
SUCCESS_MESSAGES = {
    "extraction_complete": "Text extraction completed successfully.",
    "indexing_complete": "Index building completed successfully.",
    "api_started": "API server started successfully.",
    "search_complete": "Search completed successfully.",
    "document_loaded": "Document loaded successfully.",
    "page_extracted": "Page extracted successfully.",
}

def get_config() -> Dict[str, Any]:
    """
    Get the complete configuration dictionary.
    
    Returns:
        Dict[str, Any]: Complete configuration dictionary
    """
    return {
        "paths": {
            "project_root": PROJECT_ROOT,
            "data_dir": DATA_DIR,
            "standards_dir": STANDARDS_DIR,
            "indexing_dir": INDEXING_DIR,
            "ocr_dir": OCR_DIR,
            "api_dir": API_DIR,
            "utils_dir": UTILS_DIR,
        },
        "ocr": OCR_CONFIG,
        "text_extraction": TEXT_EXTRACTION_CONFIG,
        "embedding": EMBEDDING_CONFIG,
        "faiss": FAISS_CONFIG,
        "api": API_CONFIG,
        "search": SEARCH_CONFIG,
        "logging": LOGGING_CONFIG,
        "file_paths": {k: str(v) for k, v in FILE_PATHS.items()},
        "pdf": PDF_CONFIG,
        "performance": PERFORMANCE_CONFIG,
        "dev": DEV_CONFIG,
        "models": MODEL_CONFIGS,
        "section_patterns": SECTION_PATTERNS,
        "errors": ERROR_MESSAGES,
        "success": SUCCESS_MESSAGES,
    }

def validate_config() -> bool:
    """
    Validate the current configuration.
    
    Returns:
        bool: True if configuration is valid, False otherwise
    """
    try:
        # Check required directories
        required_dirs = [STANDARDS_DIR, INDEXING_DIR]
        for dir_path in required_dirs:
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
        
        # Check model configuration
        model_name = EMBEDDING_CONFIG["model_name"]
        if model_name not in MODEL_CONFIGS:
            print(f"Warning: Model '{model_name}' not in known configurations")
        
        # Check API configuration
        if API_CONFIG["port"] < 1 or API_CONFIG["port"] > 65535:
            print("Error: Invalid API port number")
            return False
        
        # Check search configuration
        if SEARCH_CONFIG["default_top_k"] > SEARCH_CONFIG["max_top_k"]:
            print("Error: Default top_k cannot be greater than max_top_k")
            return False
        
        return True
        
    except Exception as e:
        print(f"Configuration validation failed: {e}")
        return False

# Initialize configuration validation
if __name__ == "__main__":
    if validate_config():
        print("✓ Configuration is valid")
    else:
        print("✗ Configuration validation failed")
