#!/usr/bin/env python3
"""
Complete Pipeline Runner for AI Standards Assistant

This script runs the complete pipeline from PDF processing to API server startup.
It's designed to be a one-stop solution for setting up and running the system.
"""

import os
import sys
import logging
import argparse
from pathlib import Path
import subprocess
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def check_dependencies():
    """Check if all required dependencies are installed."""
    logger.info("Checking dependencies...")
    
    required_packages = [
        'fastapi', 'uvicorn', 'pymupdf', 'pdfplumber', 'pdf2image',
        'pytesseract', 'sentence_transformers', 'faiss-cpu', 'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing packages: {', '.join(missing_packages)}")
        logger.info("Please install missing packages with: pip install -r requirements.txt")
        return False
    
    logger.info("All dependencies are installed ✓")
    return True


def check_tesseract():
    """Check if Tesseract OCR is available."""
    logger.info("Checking Tesseract OCR...")
    
    try:
        import pytesseract
        # Try to get version
        version = pytesseract.get_tesseract_version()
        logger.info(f"Tesseract version: {version} ✓")
        return True
    except Exception as e:
        logger.error(f"Tesseract OCR not found: {e}")
        logger.info("Please install Tesseract OCR:")
        logger.info("  Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
        logger.info("  macOS: brew install tesseract")
        logger.info("  Ubuntu: sudo apt install tesseract-ocr")
        return False


def check_standards_folder():
    """Check if standards folder exists and contains PDFs."""
    logger.info("Checking standards folder...")
    
    standards_folder = Path("data/standards")
    
    if not standards_folder.exists():
        logger.warning(f"Standards folder not found: {standards_folder}")
        logger.info("Creating standards folder...")
        standards_folder.mkdir(parents=True, exist_ok=True)
        logger.info("Please add your PDF files to the data/standards/ folder")
        return False
    
    pdf_files = list(standards_folder.glob("*.pdf"))
    
    if not pdf_files:
        logger.warning("No PDF files found in standards folder")
        logger.info("Please add your PDF files to the data/standards/ folder")
        return False
    
    logger.info(f"Found {len(pdf_files)} PDF files ✓")
    return True


def run_text_extraction():
    """Run the text extraction process."""
    logger.info("Starting text extraction...")
    
    try:
        # Import and run the extractor
        sys.path.append(str(Path(__file__).parent))
        from ocr.extract_text import PDFTextExtractor
        
        extractor = PDFTextExtractor()
        extractor.save_extracted_data("indexing/extracted_text.json")
        
        logger.info("Text extraction completed ✓")
        return True
        
    except Exception as e:
        logger.error(f"Text extraction failed: {e}")
        return False


def run_indexing():
    """Run the indexing process."""
    logger.info("Starting index building...")
    
    try:
        # Import and run the indexer
        sys.path.append(str(Path(__file__).parent))
        from indexing.build_index import EmbeddingIndexer
        
        indexer = EmbeddingIndexer()
        indexer.build_complete_index("indexing/extracted_text.json")
        
        logger.info("Index building completed ✓")
        return True
        
    except Exception as e:
        logger.error(f"Index building failed: {e}")
        return False


def start_api_server(host="0.0.0.0", port=8000):
    """Start the FastAPI server."""
    logger.info(f"Starting API server on {host}:{port}...")
    
    try:
        import uvicorn
        from api.app import app
        
        logger.info("API server starting...")
        logger.info(f"API documentation available at: http://{host}:{port}/docs")
        logger.info(f"Health check available at: http://{host}:{port}/health")
        
        uvicorn.run(app, host=host, port=port, log_level="info")
        
    except Exception as e:
        logger.error(f"Failed to start API server: {e}")
        return False


def main():
    """Main pipeline runner."""
    parser = argparse.ArgumentParser(description="AI Standards Assistant Pipeline Runner")
    parser.add_argument("--skip-extraction", action="store_true", 
                       help="Skip text extraction (use existing extracted_text.json)")
    parser.add_argument("--skip-indexing", action="store_true", 
                       help="Skip index building (use existing index)")
    parser.add_argument("--skip-server", action="store_true", 
                       help="Skip starting the API server")
    parser.add_argument("--host", default="0.0.0.0", 
                       help="API server host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, 
                       help="API server port (default: 8000)")
    parser.add_argument("--check-only", action="store_true", 
                       help="Only run checks, don't start the pipeline")
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("AI Standards Assistant Pipeline Runner")
    logger.info("=" * 60)
    
    # Run checks
    if not check_dependencies():
        sys.exit(1)
    
    if not check_tesseract():
        logger.warning("Tesseract not found, OCR functionality will be limited")
    
    if not check_standards_folder():
        sys.exit(1)
    
    if args.check_only:
        logger.info("All checks passed! Use --check-only=false to run the full pipeline.")
        return
    
    # Run pipeline steps
    success = True
    
    # Step 1: Text Extraction
    if not args.skip_extraction:
        if not run_text_extraction():
            success = False
    else:
        logger.info("Skipping text extraction...")
    
    # Step 2: Indexing
    if success and not args.skip_indexing:
        if not run_indexing():
            success = False
    else:
        logger.info("Skipping index building...")
    
    # Step 3: Start API Server
    if success and not args.skip_server:
        logger.info("=" * 60)
        logger.info("Starting API Server...")
        logger.info("=" * 60)
        start_api_server(args.host, args.port)
    else:
        logger.info("Skipping API server startup...")
    
    if success:
        logger.info("Pipeline completed successfully! ✓")
    else:
        logger.error("Pipeline failed! ✗")
        sys.exit(1)


if __name__ == "__main__":
    main()
