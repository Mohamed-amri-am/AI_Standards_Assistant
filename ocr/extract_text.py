"""
OCR and Text Extraction Module for AI Standards Assistant

This module handles the extraction of text from PDF documents, including:
- Detecting if a PDF page contains text or is image-based
- Performing OCR on image-based pages using Tesseract
- Extracting text from text-based pages using PyMuPDF
- Organizing extracted text by page and document
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import fitz  # PyMuPDF
import pytesseract
from pdf2image import convert_from_path
from PIL import Image
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFTextExtractor:
    """
    A class to extract text from PDF documents with OCR support for image-based pages.
    
    This class can handle both text-based and image-based PDFs by:
    1. First attempting to extract text directly from PDF pages
    2. If no text is found, converting the page to an image and performing OCR
    3. Organizing the results by document and page number
    """
    
    def __init__(self, standards_folder: str = "data/standards"):
        """
        Initialize the PDF text extractor.
        
        Args:
            standards_folder (str): Path to the folder containing PDF standards
        """
        self.standards_folder = Path(standards_folder)
        self.extracted_data = {}
        
    def has_text_content(self, page: fitz.Page) -> bool:
        """
        Check if a PDF page contains extractable text content.
        
        Args:
            page (fitz.Page): PyMuPDF page object
            
        Returns:
            bool: True if the page contains text, False otherwise
        """
        try:
            text = page.get_text().strip()
            return len(text) > 50  # Consider pages with less than 50 chars as image-based
        except Exception as e:
            logger.warning(f"Error checking text content: {e}")
            return False
    
    def extract_text_from_page(self, page: fitz.Page, page_num: int) -> str:
        """
        Extract text from a PDF page using PyMuPDF.
        
        Args:
            page (fitz.Page): PyMuPDF page object
            page_num (int): Page number (0-indexed)
            
        Returns:
            str: Extracted text from the page
        """
        try:
            text = page.get_text()
            logger.info(f"Extracted {len(text)} characters from page {page_num + 1}")
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from page {page_num + 1}: {e}")
            return ""
    
    def extract_text_with_ocr(self, pdf_path: str, page_num: int) -> str:
        """
        Extract text from a PDF page using OCR when direct text extraction fails.
        
        Args:
            pdf_path (str): Path to the PDF file
            page_num (int): Page number (0-indexed)
            
        Returns:
            str: Extracted text from the page using OCR
        """
        try:
            # Convert specific page to image
            images = convert_from_path(pdf_path, first_page=page_num + 1, last_page=page_num + 1)
            
            if not images:
                logger.warning(f"No image generated for page {page_num + 1}")
                return ""
            
            # Perform OCR on the image
            image = images[0]
            text = pytesseract.image_to_string(image, lang='eng')
            
            logger.info(f"OCR extracted {len(text)} characters from page {page_num + 1}")
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error performing OCR on page {page_num + 1}: {e}")
            return ""
    
    def process_single_pdf(self, pdf_path: Path) -> Dict:
        """
        Process a single PDF file and extract text from all pages.
        
        Args:
            pdf_path (Path): Path to the PDF file
            
        Returns:
            Dict: Dictionary containing extracted text organized by page
        """
        logger.info(f"Processing PDF: {pdf_path.name}")
        
        document_data = {
            "filename": pdf_path.name,
            "filepath": str(pdf_path),
            "pages": {},
            "total_pages": 0,
            "extraction_method": {}
        }
        
        try:
            # Open PDF with PyMuPDF
            doc = fitz.open(pdf_path)
            document_data["total_pages"] = len(doc)
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Try to extract text directly first
                if self.has_text_content(page):
                    text = self.extract_text_from_page(page, page_num)
                    extraction_method = "direct"
                else:
                    # Fall back to OCR
                    text = self.extract_text_with_ocr(str(pdf_path), page_num)
                    extraction_method = "ocr"
                
                # Store the extracted text and method used
                document_data["pages"][page_num + 1] = {
                    "text": text,
                    "extraction_method": extraction_method,
                    "character_count": len(text)
                }
                
                document_data["extraction_method"][page_num + 1] = extraction_method
                
                logger.info(f"Page {page_num + 1}: {extraction_method} extraction, {len(text)} characters")
            
            doc.close()
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path.name}: {e}")
            document_data["error"] = str(e)
        
        return document_data
    
    def extract_all_pdfs(self) -> Dict:
        """
        Extract text from all PDF files in the standards folder.
        
        Returns:
            Dict: Dictionary containing extracted text from all PDFs
        """
        if not self.standards_folder.exists():
            logger.error(f"Standards folder not found: {self.standards_folder}")
            return {}
        
        # Find all PDF files
        pdf_files = list(self.standards_folder.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.standards_folder}")
            return {}
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        all_extracted_data = {
            "extraction_timestamp": None,
            "total_documents": len(pdf_files),
            "documents": {}
        }
        
        for pdf_path in pdf_files:
            document_data = self.process_single_pdf(pdf_path)
            all_extracted_data["documents"][pdf_path.stem] = document_data
        
        # Add timestamp
        from datetime import datetime
        all_extracted_data["extraction_timestamp"] = datetime.now().isoformat()
        
        return all_extracted_data
    
    def save_extracted_data(self, output_path: str = "indexing/extracted_text.json"):
        """
        Extract text from all PDFs and save the results to a JSON file.
        
        Args:
            output_path (str): Path to save the extracted text data
        """
        logger.info("Starting text extraction from all PDFs...")
        
        extracted_data = self.extract_all_pdfs()
        
        if not extracted_data.get("documents"):
            logger.warning("No documents were processed successfully")
            return
        
        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save to JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(extracted_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Extracted text saved to: {output_path}")
        
        # Print summary
        total_pages = sum(
            doc.get("total_pages", 0) 
            for doc in extracted_data["documents"].values()
        )
        logger.info(f"Extraction complete: {len(extracted_data['documents'])} documents, {total_pages} total pages")


def main():
    """
    Main function to run the text extraction process.
    """
    extractor = PDFTextExtractor()
    extractor.save_extracted_data()


if __name__ == "__main__":
    main()
