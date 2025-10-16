#!/usr/bin/env python3
"""
System Test Script for AI Standards Assistant

This script tests the complete system functionality including:
- Text extraction
- Index building
- Search functionality
- API endpoints
"""

import os
import sys
import json
import logging
import requests
import time
from pathlib import Path
from typing import Dict, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SystemTester:
    """Test suite for the AI Standards Assistant system."""
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        """
        Initialize the system tester.
        
        Args:
            api_base_url (str): Base URL for the API server
        """
        self.api_base_url = api_base_url
        self.test_results = {}
    
    def test_text_extraction(self) -> bool:
        """Test text extraction functionality."""
        logger.info("Testing text extraction...")
        
        try:
            # Import the extractor
            sys.path.append(str(Path(__file__).parent))
            from ocr.extract_text import PDFTextExtractor
            
            # Check if standards folder exists
            standards_folder = Path("data/standards")
            if not standards_folder.exists():
                logger.warning("Standards folder not found, creating test structure...")
                standards_folder.mkdir(parents=True, exist_ok=True)
                logger.info("Please add PDF files to data/standards/ and run the test again")
                return False
            
            # Check for PDF files
            pdf_files = list(standards_folder.glob("*.pdf"))
            if not pdf_files:
                logger.warning("No PDF files found in standards folder")
                return False
            
            # Test extraction on first PDF
            extractor = PDFTextExtractor()
            test_pdf = pdf_files[0]
            result = extractor.process_single_pdf(test_pdf)
            
            if result and result.get("pages"):
                logger.info(f"✓ Text extraction successful for {test_pdf.name}")
                logger.info(f"  - Pages processed: {result.get('total_pages', 0)}")
                logger.info(f"  - Total characters: {sum(page.get('character_count', 0) for page in result.get('pages', {}).values())}")
                return True
            else:
                logger.error("✗ Text extraction failed")
                return False
                
        except Exception as e:
            logger.error(f"✗ Text extraction test failed: {e}")
            return False
    
    def test_indexing(self) -> bool:
        """Test indexing functionality."""
        logger.info("Testing indexing...")
        
        try:
            # Check if extracted text exists
            extracted_file = Path("indexing/extracted_text.json")
            if not extracted_file.exists():
                logger.warning("Extracted text file not found, running extraction first...")
                if not self.test_text_extraction():
                    return False
            
            # Import the indexer
            sys.path.append(str(Path(__file__).parent))
            from indexing.build_index import EmbeddingIndexer
            
            # Test indexing
            indexer = EmbeddingIndexer()
            segments = indexer.process_extracted_data("indexing/extracted_text.json")
            
            if segments:
                logger.info(f"✓ Indexing successful")
                logger.info(f"  - Segments created: {len(segments)}")
                
                # Test embedding creation
                embeddings = indexer.create_embeddings(segments[:10])  # Test with first 10 segments
                logger.info(f"  - Embeddings shape: {embeddings.shape}")
                
                # Test FAISS index
                faiss_index = indexer.build_faiss_index(embeddings)
                logger.info(f"  - FAISS index size: {faiss_index.ntotal}")
                
                return True
            else:
                logger.error("✗ No segments created during indexing")
                return False
                
        except Exception as e:
            logger.error(f"✗ Indexing test failed: {e}")
            return False
    
    def test_search_functionality(self) -> bool:
        """Test search functionality."""
        logger.info("Testing search functionality...")
        
        try:
            # Import the indexer
            sys.path.append(str(Path(__file__).parent))
            from indexing.build_index import EmbeddingIndexer
            
            # Load existing index
            indexer = EmbeddingIndexer()
            indexer.load_index("indexing")
            
            # Test queries
            test_queries = [
                "safety requirements",
                "noise level",
                "electrical equipment",
                "fire prevention"
            ]
            
            for query in test_queries:
                results = indexer.search(query, top_k=3)
                if results:
                    logger.info(f"✓ Search successful for '{query}'")
                    logger.info(f"  - Results found: {len(results)}")
                    logger.info(f"  - Top score: {results[0]['similarity_score']:.3f}")
                else:
                    logger.warning(f"⚠ No results for query '{query}'")
            
            return True
            
        except Exception as e:
            logger.error(f"✗ Search functionality test failed: {e}")
            return False
    
    def test_api_health(self) -> bool:
        """Test API health endpoint."""
        logger.info("Testing API health...")
        
        try:
            response = requests.get(f"{self.api_base_url}/health", timeout=10)
            
            if response.status_code == 200:
                health_data = response.json()
                logger.info("✓ API health check passed")
                logger.info(f"  - Status: {health_data.get('status')}")
                logger.info(f"  - Index loaded: {health_data.get('index_loaded')}")
                logger.info(f"  - Total segments: {health_data.get('total_segments')}")
                return True
            else:
                logger.error(f"✗ API health check failed: {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"✗ API health check failed: {e}")
            return False
    
    def test_api_search(self) -> bool:
        """Test API search endpoint."""
        logger.info("Testing API search...")
        
        try:
            # Test POST search
            search_data = {
                "query": "safety requirements",
                "top_k": 3,
                "include_preview": False
            }
            
            response = requests.post(
                f"{self.api_base_url}/search",
                json=search_data,
                timeout=30
            )
            
            if response.status_code == 200:
                search_results = response.json()
                logger.info("✓ API search successful")
                logger.info(f"  - Query: {search_results.get('query')}")
                logger.info(f"  - Results: {search_results.get('total_results')}")
                logger.info(f"  - Processing time: {search_results.get('processing_time_ms'):.1f}ms")
                
                # Test GET search
                get_response = requests.get(
                    f"{self.api_base_url}/search",
                    params={"q": "noise level", "top_k": 2},
                    timeout=30
                )
                
                if get_response.status_code == 200:
                    logger.info("✓ API GET search successful")
                    return True
                else:
                    logger.error(f"✗ API GET search failed: {get_response.status_code}")
                    return False
            else:
                logger.error(f"✗ API search failed: {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"✗ API search test failed: {e}")
            return False
    
    def test_api_documents(self) -> bool:
        """Test API documents endpoint."""
        logger.info("Testing API documents...")
        
        try:
            response = requests.get(f"{self.api_base_url}/documents", timeout=10)
            
            if response.status_code == 200:
                docs_data = response.json()
                logger.info("✓ API documents endpoint successful")
                logger.info(f"  - Total documents: {docs_data.get('total_documents')}")
                
                # Test individual document info
                documents = docs_data.get('documents', [])
                if documents:
                    doc_name = documents[0]['name']
                    info_response = requests.get(
                        f"{self.api_base_url}/documents/{doc_name}/info",
                        timeout=10
                    )
                    
                    if info_response.status_code == 200:
                        logger.info(f"✓ Document info successful for {doc_name}")
                        return True
                    else:
                        logger.error(f"✗ Document info failed: {info_response.status_code}")
                        return False
                
                return True
            else:
                logger.error(f"✗ API documents failed: {response.status_code}")
                return False
                
        except requests.exceptions.RequestException as e:
            logger.error(f"✗ API documents test failed: {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all tests and return results."""
        logger.info("=" * 60)
        logger.info("Running AI Standards Assistant System Tests")
        logger.info("=" * 60)
        
        tests = [
            ("Text Extraction", self.test_text_extraction),
            ("Indexing", self.test_indexing),
            ("Search Functionality", self.test_search_functionality),
            ("API Health", self.test_api_health),
            ("API Search", self.test_api_search),
            ("API Documents", self.test_api_documents)
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            logger.info(f"\n--- Running {test_name} Test ---")
            try:
                results[test_name] = test_func()
            except Exception as e:
                logger.error(f"✗ {test_name} test crashed: {e}")
                results[test_name] = False
        
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("Test Results Summary")
        logger.info("=" * 60)
        
        passed = 0
        total = len(results)
        
        for test_name, result in results.items():
            status = "✓ PASS" if result else "✗ FAIL"
            logger.info(f"{test_name:20} {status}")
            if result:
                passed += 1
        
        logger.info(f"\nOverall: {passed}/{total} tests passed")
        
        if passed == total:
            logger.info("🎉 All tests passed! System is working correctly.")
        else:
            logger.warning(f"⚠️  {total - passed} tests failed. Please check the issues above.")
        
        return results


def main():
    """Main test runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test AI Standards Assistant System")
    parser.add_argument("--api-url", default="http://localhost:8000",
                       help="API server URL (default: http://localhost:8000)")
    parser.add_argument("--wait-for-api", action="store_true",
                       help="Wait for API server to be available")
    
    args = parser.parse_args()
    
    tester = SystemTester(args.api_url)
    
    # Wait for API if requested
    if args.wait_for_api:
        logger.info("Waiting for API server to be available...")
        max_attempts = 30
        for attempt in range(max_attempts):
            try:
                response = requests.get(f"{args.api_url}/health", timeout=5)
                if response.status_code == 200:
                    logger.info("API server is available!")
                    break
            except:
                pass
            
            if attempt < max_attempts - 1:
                time.sleep(2)
                logger.info(f"Attempt {attempt + 1}/{max_attempts}...")
        else:
            logger.error("API server not available after waiting")
            sys.exit(1)
    
    # Run tests
    results = tester.run_all_tests()
    
    # Exit with appropriate code
    if all(results.values()):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
