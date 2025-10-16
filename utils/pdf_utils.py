"""
PDF Utilities Module for AI Standards Assistant

This module provides utility functions for working with PDF documents, including:
- Extracting specific pages as images
- Finding text snippets within pages
- Converting PDF pages to images for preview
- Handling PDF metadata and structure
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import fitz  # PyMuPDF
from PIL import Image
import io
import base64

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFUtils:
    """
    A utility class for PDF operations including page extraction and image conversion.
    
    This class provides methods to extract specific pages from PDFs, convert them
    to images, and find text snippets for preview purposes.
    """
    
    def __init__(self, standards_folder: str = "data/standards"):
        """
        Initialize the PDF utilities.
        
        Args:
            standards_folder (str): Path to the folder containing PDF standards
        """
        self.standards_folder = Path(standards_folder)
    
    def get_pdf_path(self, document_name: str) -> Optional[Path]:
        """
        Get the full path to a PDF document.
        
        Args:
            document_name (str): Name of the document (without extension)
            
        Returns:
            Optional[Path]: Path to the PDF file if found, None otherwise
        """
        # Try different possible extensions and naming conventions
        possible_names = [
            f"{document_name}.pdf",
            f"{document_name}.PDF",
            document_name  # In case the name already includes extension
        ]
        
        for name in possible_names:
            pdf_path = self.standards_folder / name
            if pdf_path.exists():
                return pdf_path
        
        # If not found, search for files that contain the document name
        for pdf_file in self.standards_folder.glob("*.pdf"):
            if document_name.lower() in pdf_file.stem.lower():
                return pdf_file
        
        logger.warning(f"PDF not found for document: {document_name}")
        return None
    
    def extract_page_as_image(self, document_name: str, page_num: int, 
                            dpi: int = 150, format: str = "PNG") -> Optional[bytes]:
        """
        Extract a specific page from a PDF as an image.
        
        Args:
            document_name (str): Name of the document
            page_num (int): Page number (1-indexed)
            dpi (int): Resolution for the image
            format (str): Image format (PNG, JPEG, etc.)
            
        Returns:
            Optional[bytes]: Image data as bytes, None if extraction fails
        """
        pdf_path = self.get_pdf_path(document_name)
        if not pdf_path:
            return None
        
        try:
            # Open PDF
            doc = fitz.open(pdf_path)
            
            # Check if page number is valid
            if page_num < 1 or page_num > len(doc):
                logger.error(f"Invalid page number {page_num} for document {document_name}")
                doc.close()
                return None
            
            # Get the page (convert to 0-indexed)
            page = doc[page_num - 1]
            
            # Create transformation matrix for DPI
            mat = fitz.Matrix(dpi / 72, dpi / 72)
            
            # Render page as image
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to bytes
            img_data = pix.tobytes(format.lower())
            
            doc.close()
            
            logger.info(f"Extracted page {page_num} from {document_name} as {format}")
            return img_data
            
        except Exception as e:
            logger.error(f"Error extracting page {page_num} from {document_name}: {e}")
            return None
    
    def extract_page_as_base64(self, document_name: str, page_num: int, 
                             dpi: int = 150, format: str = "PNG") -> Optional[str]:
        """
        Extract a specific page from a PDF as a base64-encoded image.
        
        Args:
            document_name (str): Name of the document
            page_num (int): Page number (1-indexed)
            dpi (int): Resolution for the image
            format (str): Image format (PNG, JPEG, etc.)
            
        Returns:
            Optional[str]: Base64-encoded image data, None if extraction fails
        """
        img_data = self.extract_page_as_image(document_name, page_num, dpi, format)
        if img_data:
            return base64.b64encode(img_data).decode('utf-8')
        return None
    
    def find_text_in_page(self, document_name: str, page_num: int, 
                         search_text: str, case_sensitive: bool = False) -> List[Dict]:
        """
        Find occurrences of text within a specific page.
        
        Args:
            document_name (str): Name of the document
            page_num (int): Page number (1-indexed)
            search_text (str): Text to search for
            case_sensitive (bool): Whether search should be case sensitive
            
        Returns:
            List[Dict]: List of text matches with positions and context
        """
        pdf_path = self.get_pdf_path(document_name)
        if not pdf_path:
            return []
        
        try:
            # Open PDF
            doc = fitz.open(pdf_path)
            
            # Check if page number is valid
            if page_num < 1 or page_num > len(doc):
                logger.error(f"Invalid page number {page_num} for document {document_name}")
                doc.close()
                return []
            
            # Get the page (convert to 0-indexed)
            page = doc[page_num - 1]
            
            # Search for text
            text_instances = page.search_for(search_text, flags=fitz.TEXTFLAGS_TEXT)
            
            matches = []
            for inst in text_instances:
                # Get text around the match for context
                rect = fitz.Rect(inst)
                
                # Expand rectangle to get more context
                expanded_rect = fitz.Rect(
                    rect.x0 - 50, rect.y0 - 20,
                    rect.x1 + 50, rect.y1 + 20
                )
                
                # Get text from expanded area
                context_text = page.get_textbox(expanded_rect)
                
                matches.append({
                    'bbox': [rect.x0, rect.y0, rect.x1, rect.y1],
                    'context': context_text.strip(),
                    'page': page_num,
                    'document': document_name
                })
            
            doc.close()
            
            logger.info(f"Found {len(matches)} matches for '{search_text}' in page {page_num} of {document_name}")
            return matches
            
        except Exception as e:
            logger.error(f"Error searching text in page {page_num} of {document_name}: {e}")
            return []
    
    def get_page_text(self, document_name: str, page_num: int) -> Optional[str]:
        """
        Get all text from a specific page.
        
        Args:
            document_name (str): Name of the document
            page_num (int): Page number (1-indexed)
            
        Returns:
            Optional[str]: Text content of the page, None if extraction fails
        """
        pdf_path = self.get_pdf_path(document_name)
        if not pdf_path:
            return None
        
        try:
            # Open PDF
            doc = fitz.open(pdf_path)
            
            # Check if page number is valid
            if page_num < 1 or page_num > len(doc):
                logger.error(f"Invalid page number {page_num} for document {document_name}")
                doc.close()
                return None
            
            # Get the page (convert to 0-indexed)
            page = doc[page_num - 1]
            
            # Extract text
            text = page.get_text()
            
            doc.close()
            
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error extracting text from page {page_num} of {document_name}: {e}")
            return None
    
    def get_document_info(self, document_name: str) -> Optional[Dict]:
        """
        Get metadata information about a PDF document.
        
        Args:
            document_name (str): Name of the document
            
        Returns:
            Optional[Dict]: Document metadata, None if document not found
        """
        pdf_path = self.get_pdf_path(document_name)
        if not pdf_path:
            return None
        
        try:
            # Open PDF
            doc = fitz.open(pdf_path)
            
            # Get metadata
            metadata = doc.metadata
            
            # Get additional info
            info = {
                'filename': pdf_path.name,
                'filepath': str(pdf_path),
                'total_pages': len(doc),
                'title': metadata.get('title', ''),
                'author': metadata.get('author', ''),
                'subject': metadata.get('subject', ''),
                'creator': metadata.get('creator', ''),
                'producer': metadata.get('producer', ''),
                'creation_date': metadata.get('creationDate', ''),
                'modification_date': metadata.get('modDate', '')
            }
            
            doc.close()
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting document info for {document_name}: {e}")
            return None
    
    def highlight_text_in_page(self, document_name: str, page_num: int, 
                              search_text: str, highlight_color: Tuple[float, float, float] = (1, 1, 0)) -> Optional[bytes]:
        """
        Create an image of a page with highlighted text.
        
        Args:
            document_name (str): Name of the document
            page_num (int): Page number (1-indexed)
            search_text (str): Text to highlight
            highlight_color (Tuple[float, float, float]): RGB color for highlighting (0-1 range)
            
        Returns:
            Optional[bytes]: Image data with highlighted text, None if extraction fails
        """
        pdf_path = self.get_pdf_path(document_name)
        if not pdf_path:
            return None
        
        try:
            # Open PDF
            doc = fitz.open(pdf_path)
            
            # Check if page number is valid
            if page_num < 1 or page_num > len(doc):
                logger.error(f"Invalid page number {page_num} for document {document_name}")
                doc.close()
                return None
            
            # Get the page (convert to 0-indexed)
            page = doc[page_num - 1]
            
            # Search for text and highlight
            text_instances = page.search_for(search_text)
            
            # Add highlights
            for inst in text_instances:
                highlight = page.add_highlight_annot(inst)
                highlight.set_colors(stroke=highlight_color)
                highlight.update()
            
            # Render page as image
            mat = fitz.Matrix(150 / 72, 150 / 72)  # 150 DPI
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            
            doc.close()
            
            logger.info(f"Created highlighted image for page {page_num} of {document_name}")
            return img_data
            
        except Exception as e:
            logger.error(f"Error creating highlighted image for page {page_num} of {document_name}: {e}")
            return None
    
    def get_text_snippet(self, document_name: str, page_num: int, 
                        start_pos: int = 0, length: int = 500) -> Optional[str]:
        """
        Get a text snippet from a specific position in a page.
        
        Args:
            document_name (str): Name of the document
            page_num (int): Page number (1-indexed)
            start_pos (int): Starting character position
            length (int): Length of the snippet
            
        Returns:
            Optional[str]: Text snippet, None if extraction fails
        """
        full_text = self.get_page_text(document_name, page_num)
        if not full_text:
            return None
        
        # Ensure start_pos is within bounds
        start_pos = max(0, min(start_pos, len(full_text)))
        
        # Get snippet
        snippet = full_text[start_pos:start_pos + length]
        
        # Try to start at a word boundary
        if start_pos > 0 and snippet and not snippet[0].isspace():
            # Find the start of the current word
            while start_pos > 0 and not full_text[start_pos - 1].isspace():
                start_pos -= 1
            snippet = full_text[start_pos:start_pos + length]
        
        return snippet.strip()


def create_preview_image(search_result: Dict, pdf_utils: PDFUtils, 
                        highlight_text: Optional[str] = None) -> Optional[str]:
    """
    Create a preview image for a search result.
    
    Args:
        search_result (Dict): Search result from the index
        pdf_utils (PDFUtils): PDFUtils instance
        highlight_text (Optional[str]): Text to highlight in the image
        
    Returns:
        Optional[str]: Base64-encoded image data, None if creation fails
    """
    document_name = search_result.get('document', '')
    page_num = search_result.get('page', 1)
    
    if highlight_text:
        # Create highlighted image
        img_data = pdf_utils.highlight_text_in_page(document_name, page_num, highlight_text)
    else:
        # Create regular image
        img_data = pdf_utils.extract_page_as_image(document_name, page_num)
    
    if img_data:
        return base64.b64encode(img_data).decode('utf-8')
    
    return None


def main():
    """
    Main function for testing PDF utilities.
    """
    pdf_utils = PDFUtils()
    
    # List all PDFs in the standards folder
    pdf_files = list(pdf_utils.standards_folder.glob("*.pdf"))
    
    if not pdf_files:
        logger.info("No PDF files found in standards folder")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF files:")
    for pdf_file in pdf_files:
        logger.info(f"  - {pdf_file.name}")
        
        # Get document info
        doc_name = pdf_file.stem
        info = pdf_utils.get_document_info(doc_name)
        if info:
            logger.info(f"    Pages: {info['total_pages']}")
            logger.info(f"    Title: {info['title']}")


if __name__ == "__main__":
    main()
