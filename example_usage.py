#!/usr/bin/env python3
"""
Example Usage Script for AI Standards Assistant

This script demonstrates how to use the AI Standards Assistant system
programmatically without the API server.
"""

import sys
import json
from pathlib import Path
from typing import List, Dict

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from indexing.build_index import EmbeddingIndexer
from utils.pdf_utils import PDFUtils, create_preview_image


def example_search():
    """Example of how to search the standards without the API."""
    print("=" * 60)
    print("AI Standards Assistant - Direct Usage Example")
    print("=" * 60)
    
    # Initialize the indexer
    print("Loading search index...")
    indexer = EmbeddingIndexer()
    
    try:
        indexer.load_index("indexing")
        print(f"✓ Index loaded with {len(indexer.metadata)} segments")
    except FileNotFoundError:
        print("✗ Index not found. Please run the indexing process first:")
        print("  1. python ocr/extract_text.py")
        print("  2. python indexing/build_index.py")
        return
    
    # Initialize PDF utils
    print("Initializing PDF utilities...")
    pdf_utils = PDFUtils()
    
    # Example queries
    queries = [
        "noise level in workshop exceeds the limit",
        "safety requirements for electrical equipment",
        "fire prevention measures in industrial facilities",
        "maximum temperature limits for machinery",
        "personal protective equipment requirements"
    ]
    
    print("\n" + "=" * 60)
    print("Searching Standards Documents")
    print("=" * 60)
    
    for i, query in enumerate(queries, 1):
        print(f"\n--- Query {i}: {query} ---")
        
        # Perform search
        results = indexer.search(query, top_k=3)
        
        if results:
            print(f"Found {len(results)} relevant sections:")
            
            for j, result in enumerate(results, 1):
                print(f"\n  Result {j} (Score: {result['similarity_score']:.3f}):")
                print(f"    Document: {result['document']}")
                print(f"    Page: {result['page']}")
                if result.get('header'):
                    print(f"    Section: {result['header']}")
                print(f"    Text: {result['text'][:200]}...")
                
                # Optional: Create preview image
                try:
                    preview = create_preview_image(result, pdf_utils)
                    if preview:
                        print(f"    Preview: Generated (base64 length: {len(preview)})")
                except Exception as e:
                    print(f"    Preview: Failed to generate ({e})")
        else:
            print("No relevant sections found.")
    
    print("\n" + "=" * 60)
    print("Search Complete!")
    print("=" * 60)


def example_document_info():
    """Example of how to get document information."""
    print("\n" + "=" * 60)
    print("Document Information Example")
    print("=" * 60)
    
    pdf_utils = PDFUtils()
    
    # List all documents
    pdf_files = list(pdf_utils.standards_folder.glob("*.pdf"))
    
    if not pdf_files:
        print("No PDF files found in data/standards/")
        return
    
    print(f"Found {len(pdf_files)} documents:")
    
    for pdf_file in pdf_files:
        doc_name = pdf_file.stem
        info = pdf_utils.get_document_info(doc_name)
        
        if info:
            print(f"\n📄 {doc_name}")
            print(f"   Pages: {info['total_pages']}")
            print(f"   Title: {info['title'] or 'N/A'}")
            print(f"   Author: {info['author'] or 'N/A'}")
            print(f"   File size: {pdf_file.stat().st_size / 1024 / 1024:.1f} MB")


def example_page_extraction():
    """Example of how to extract specific pages."""
    print("\n" + "=" * 60)
    print("Page Extraction Example")
    print("=" * 60)
    
    pdf_utils = PDFUtils()
    
    # Get first available document
    pdf_files = list(pdf_utils.standards_folder.glob("*.pdf"))
    
    if not pdf_files:
        print("No PDF files found in data/standards/")
        return
    
    doc_name = pdf_files[0].stem
    print(f"Extracting pages from: {doc_name}")
    
    # Get document info
    info = pdf_utils.get_document_info(doc_name)
    if not info:
        print("Could not get document info")
        return
    
    total_pages = info['total_pages']
    print(f"Total pages: {total_pages}")
    
    # Extract first few pages as examples
    for page_num in range(1, min(4, total_pages + 1)):
        print(f"\n--- Page {page_num} ---")
        
        # Get text
        text = pdf_utils.get_page_text(doc_name, page_num)
        if text:
            print(f"Text length: {len(text)} characters")
            print(f"Preview: {text[:150]}...")
        
        # Get image (base64)
        img_data = pdf_utils.extract_page_as_base64(doc_name, page_num)
        if img_data:
            print(f"Image: Generated (base64 length: {len(img_data)})")


def main():
    """Main example runner."""
    print("AI Standards Assistant - Example Usage")
    print("This script demonstrates direct usage of the system components.")
    
    # Check if we have the necessary files
    if not Path("indexing/faiss_index.bin").exists():
        print("\n⚠️  Index files not found. Please run the setup process first:")
        print("   1. Add PDF files to data/standards/")
        print("   2. python ocr/extract_text.py")
        print("   3. python indexing/build_index.py")
        print("\nFor now, showing document info example only...")
        example_document_info()
        return
    
    # Run examples
    example_search()
    example_document_info()
    example_page_extraction()
    
    print("\n" + "=" * 60)
    print("Examples Complete!")
    print("=" * 60)
    print("\nTo use the API server instead:")
    print("  1. python api/app.py")
    print("  2. Visit http://localhost:8000/docs for interactive API")
    print("  3. Use the /search endpoint with your queries")


if __name__ == "__main__":
    main()
