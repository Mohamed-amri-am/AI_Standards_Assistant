"""
FastAPI Application for AI Standards Assistant

This module provides a REST API for querying the AI Standards Assistant system.
It allows users to search through standards documents using natural language queries
and retrieve relevant sections with optional page previews.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Import our modules
import sys
sys.path.append(str(Path(__file__).parent.parent))

from indexing.build_index import EmbeddingIndexer
from utils.pdf_utils import PDFUtils, create_preview_image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for the indexer and PDF utils
indexer: Optional[EmbeddingIndexer] = None
pdf_utils: Optional[PDFUtils] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager to initialize and cleanup resources.
    """
    # Startup
    logger.info("Starting AI Standards Assistant API...")
    
    global indexer, pdf_utils
    
    try:
        # Initialize PDF utils
        pdf_utils = PDFUtils()
        logger.info("PDF utils initialized")
        
        # Load the embedding index
        indexer = EmbeddingIndexer()
        indexer.load_index("indexing")
        logger.info("Embedding index loaded successfully")
        
        logger.info("AI Standards Assistant API started successfully!")
        
    except Exception as e:
        logger.error(f"Failed to initialize API: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down AI Standards Assistant API...")


# Create FastAPI app
app = FastAPI(
    title="AI Standards Assistant API",
    description="A semantic search API for standards documents using AI embeddings",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request/response
class SearchQuery(BaseModel):
    """Model for search query requests."""
    query: str = Field(..., description="Natural language query to search for", min_length=1)
    top_k: int = Field(default=5, description="Number of top results to return", ge=1, le=20)
    include_preview: bool = Field(default=False, description="Whether to include page preview images")
    highlight_text: Optional[str] = Field(default=None, description="Text to highlight in preview images")


class SearchResult(BaseModel):
    """Model for individual search results."""
    rank: int = Field(..., description="Rank of the result (1-based)")
    similarity_score: float = Field(..., description="Similarity score (0-1)")
    document: str = Field(..., description="Name of the document")
    page: int = Field(..., description="Page number")
    header: Optional[str] = Field(None, description="Section header if available")
    text: str = Field(..., description="Relevant text content")
    text_length: int = Field(..., description="Length of the text content")
    preview_image: Optional[str] = Field(None, description="Base64-encoded preview image")


class SearchResponse(BaseModel):
    """Model for search response."""
    query: str = Field(..., description="Original query")
    total_results: int = Field(..., description="Total number of results found")
    results: List[SearchResult] = Field(..., description="List of search results")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class HealthResponse(BaseModel):
    """Model for health check response."""
    status: str = Field(..., description="Health status")
    index_loaded: bool = Field(..., description="Whether the index is loaded")
    total_segments: int = Field(..., description="Total number of indexed segments")
    documents_available: int = Field(..., description="Number of documents available")


class IndexStatus(BaseModel):
    """Model for index status response."""
    index_exists: bool = Field(..., description="Whether the index exists")
    total_segments: int = Field(..., description="Total number of segments")
    model_name: str = Field(..., description="Model used for embeddings")
    last_updated: Optional[str] = Field(None, description="Last update timestamp")


# API Routes
@app.get("/", response_model=Dict[str, str])
async def root():
    """
    Root endpoint providing API information.
    """
    return {
        "message": "AI Standards Assistant API",
        "version": "1.0.0",
        "description": "Semantic search for standards documents",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint to verify API and index status.
    """
    global indexer, pdf_utils
    
    if indexer is None or pdf_utils is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    # Count available documents
    pdf_files = list(pdf_utils.standards_folder.glob("*.pdf"))
    
    return HealthResponse(
        status="healthy",
        index_loaded=indexer.index is not None,
        total_segments=len(indexer.metadata) if indexer.metadata else 0,
        documents_available=len(pdf_files)
    )


@app.get("/index/status", response_model=IndexStatus)
async def get_index_status():
    """
    Get information about the current index status.
    """
    global indexer
    
    if indexer is None:
        raise HTTPException(status_code=503, detail="Index not loaded")
    
    # Check if index files exist
    index_dir = Path("indexing")
    index_exists = (
        (index_dir / "faiss_index.bin").exists() and
        (index_dir / "standards_data.json").exists()
    )
    
    # Get last update time
    last_updated = None
    if index_exists:
        try:
            import json
            with open(index_dir / "index_info.json", 'r') as f:
                info = json.load(f)
                last_updated = info.get("last_updated")
        except:
            pass
    
    return IndexStatus(
        index_exists=index_exists,
        total_segments=len(indexer.metadata) if indexer.metadata else 0,
        model_name=indexer.model_name,
        last_updated=last_updated
    )


@app.post("/search", response_model=SearchResponse)
async def search_standards(search_query: SearchQuery):
    """
    Search for relevant sections in standards documents.
    
    This endpoint accepts a natural language query and returns the most relevant
    sections from the indexed standards documents.
    """
    global indexer, pdf_utils
    
    if indexer is None or pdf_utils is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    import time
    start_time = time.time()
    
    try:
        # Perform the search
        results = indexer.search(search_query.query, search_query.top_k)
        
        # Process results
        processed_results = []
        for result in results:
            # Create preview image if requested
            preview_image = None
            if search_query.include_preview:
                preview_image = create_preview_image(
                    result, 
                    pdf_utils, 
                    search_query.highlight_text
                )
            
            # Create search result object
            search_result = SearchResult(
                rank=result['rank'],
                similarity_score=result['similarity_score'],
                document=result['document'],
                page=result['page'],
                header=result.get('header'),
                text=result['text'],
                text_length=result['length'],
                preview_image=preview_image
            )
            
            processed_results.append(search_result)
        
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        return SearchResponse(
            query=search_query.query,
            total_results=len(processed_results),
            results=processed_results,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error during search: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.get("/search", response_model=SearchResponse)
async def search_standards_get(
    q: str = Query(..., description="Natural language query to search for"),
    top_k: int = Query(default=5, description="Number of top results to return", ge=1, le=20),
    include_preview: bool = Query(default=False, description="Whether to include page preview images"),
    highlight_text: Optional[str] = Query(default=None, description="Text to highlight in preview images")
):
    """
    GET version of the search endpoint for convenience.
    """
    search_query = SearchQuery(
        query=q,
        top_k=top_k,
        include_preview=include_preview,
        highlight_text=highlight_text
    )
    
    return await search_standards(search_query)


@app.get("/documents")
async def list_documents():
    """
    List all available documents in the standards folder.
    """
    global pdf_utils
    
    if pdf_utils is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    pdf_files = list(pdf_utils.standards_folder.glob("*.pdf"))
    
    documents = []
    for pdf_file in pdf_files:
        doc_name = pdf_file.stem
        info = pdf_utils.get_document_info(doc_name)
        
        if info:
            documents.append({
                "name": doc_name,
                "filename": info["filename"],
                "total_pages": info["total_pages"],
                "title": info["title"],
                "author": info["author"],
                "file_size": pdf_file.stat().st_size
            })
    
    return {
        "total_documents": len(documents),
        "documents": documents
    }


@app.get("/documents/{document_name}/info")
async def get_document_info(document_name: str):
    """
    Get detailed information about a specific document.
    """
    global pdf_utils
    
    if pdf_utils is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    info = pdf_utils.get_document_info(document_name)
    
    if not info:
        raise HTTPException(status_code=404, detail=f"Document '{document_name}' not found")
    
    return info


@app.get("/documents/{document_name}/pages/{page_num}/text")
async def get_page_text(document_name: str, page_num: int):
    """
    Get the text content of a specific page.
    """
    global pdf_utils
    
    if pdf_utils is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    text = pdf_utils.get_page_text(document_name, page_num)
    
    if text is None:
        raise HTTPException(
            status_code=404, 
            detail=f"Page {page_num} not found in document '{document_name}'"
        )
    
    return {
        "document": document_name,
        "page": page_num,
        "text": text,
        "length": len(text)
    }


@app.get("/documents/{document_name}/pages/{page_num}/image")
async def get_page_image(
    document_name: str, 
    page_num: int,
    dpi: int = Query(default=150, description="Image resolution", ge=72, le=300)
):
    """
    Get a specific page as an image.
    """
    global pdf_utils
    
    if pdf_utils is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    img_data = pdf_utils.extract_page_as_base64(document_name, page_num, dpi)
    
    if not img_data:
        raise HTTPException(
            status_code=404, 
            detail=f"Page {page_num} not found in document '{document_name}'"
        )
    
    return {
        "document": document_name,
        "page": page_num,
        "image_data": img_data,
        "format": "PNG",
        "dpi": dpi
    }


@app.post("/rebuild-index")
async def rebuild_index(background_tasks: BackgroundTasks):
    """
    Rebuild the search index from scratch.
    
    This endpoint triggers a background task to rebuild the entire index.
    """
    global indexer
    
    if indexer is None:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    # Add background task to rebuild index
    background_tasks.add_task(rebuild_index_task)
    
    return {
        "message": "Index rebuild started in background",
        "status": "processing"
    }


async def rebuild_index_task():
    """
    Background task to rebuild the index.
    """
    global indexer
    
    try:
        logger.info("Starting index rebuild...")
        
        # Import here to avoid circular imports
        from ocr.extract_text import PDFTextExtractor
        
        # Extract text from all PDFs
        extractor = PDFTextExtractor()
        extractor.save_extracted_data("indexing/extracted_text.json")
        
        # Rebuild index
        indexer.build_complete_index("indexing/extracted_text.json")
        
        logger.info("Index rebuild completed successfully!")
        
    except Exception as e:
        logger.error(f"Index rebuild failed: {e}")


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Resource not found"}
    )


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )


def main():
    """
    Main function to run the FastAPI server.
    """
    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )


if __name__ == "__main__":
    main()
