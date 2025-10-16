"""
Indexing Module for AI Standards Assistant

This module handles the creation of semantic embeddings and FAISS index for the extracted text.
It segments documents into meaningful sections, generates embeddings using sentence-transformers,
and creates a searchable FAISS index for efficient similarity search.
"""

import os
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentSegmenter:
    """
    A class to segment PDF documents into meaningful sections for better search results.
    
    This class identifies chapter boundaries, section headers, and other structural
    elements to create more focused and relevant text segments.
    """
    
    def __init__(self):
        """Initialize the document segmenter with common patterns."""
        # Common patterns for section headers
        self.section_patterns = [
            r'^\s*(?:Chapter|CHAPTER)\s+(\d+[\.\-\s]*[A-Za-z\s]+)',
            r'^\s*(\d+\.\d*)\s+([A-Z][A-Za-z\s]+)',
            r'^\s*([A-Z][A-Z\s]{3,})',
            r'^\s*(?:Section|SECTION)\s+(\d+[\.\-\s]*[A-Za-z\s]+)',
            r'^\s*(\d+\.\d+\.\d*)\s+([A-Z][A-Za-z\s]+)',
            r'^\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s*$'
        ]
        
        # Minimum section length (in characters)
        self.min_section_length = 200
        
        # Maximum section length (in characters)
        self.max_section_length = 2000
    
    def is_section_header(self, line: str) -> bool:
        """
        Check if a line appears to be a section header.
        
        Args:
            line (str): Line of text to check
            
        Returns:
            bool: True if the line appears to be a section header
        """
        line = line.strip()
        if len(line) < 3 or len(line) > 100:
            return False
        
        # Check against known patterns
        for pattern in self.section_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                return True
        
        # Check for all caps (common in standards)
        if line.isupper() and len(line.split()) <= 8:
            return True
        
        # Check for numbered sections
        if re.match(r'^\s*\d+\.\d*\s+', line):
            return True
        
        return False
    
    def segment_text(self, text: str, page_num: int, document_name: str) -> List[Dict]:
        """
        Segment text into meaningful sections.
        
        Args:
            text (str): Text to segment
            page_num (int): Page number
            document_name (str): Name of the document
            
        Returns:
            List[Dict]: List of text segments with metadata
        """
        if not text or len(text.strip()) < self.min_section_length:
            return []
        
        lines = text.split('\n')
        segments = []
        current_segment = []
        current_header = None
        
        for line in lines:
            line = line.strip()
            
            # Check if this line is a section header
            if self.is_section_header(line):
                # Save current segment if it has enough content
                if current_segment and len(' '.join(current_segment)) >= self.min_section_length:
                    segment_text = ' '.join(current_segment)
                    if len(segment_text) <= self.max_section_length:
                        segments.append({
                            'text': segment_text,
                            'header': current_header,
                            'page': page_num,
                            'document': document_name,
                            'length': len(segment_text)
                        })
                    else:
                        # Split long segments
                        segments.extend(self._split_long_segment(segment_text, current_header, page_num, document_name))
                
                # Start new segment
                current_segment = [line]
                current_header = line
            else:
                # Add line to current segment
                if line:  # Skip empty lines
                    current_segment.append(line)
        
        # Don't forget the last segment
        if current_segment and len(' '.join(current_segment)) >= self.min_section_length:
            segment_text = ' '.join(current_segment)
            if len(segment_text) <= self.max_section_length:
                segments.append({
                    'text': segment_text,
                    'header': current_header,
                    'page': page_num,
                    'document': document_name,
                    'length': len(segment_text)
                })
            else:
                segments.extend(self._split_long_segment(segment_text, current_header, page_num, document_name))
        
        # If no segments were created, create one from the entire text
        if not segments and text:
            full_text = text.strip()
            if len(full_text) >= self.min_section_length:
                if len(full_text) <= self.max_section_length:
                    segments.append({
                        'text': full_text,
                        'header': None,
                        'page': page_num,
                        'document': document_name,
                        'length': len(full_text)
                    })
                else:
                    segments.extend(self._split_long_segment(full_text, None, page_num, document_name))
        
        return segments
    
    def _split_long_segment(self, text: str, header: Optional[str], page_num: int, document_name: str) -> List[Dict]:
        """
        Split a long text segment into smaller chunks.
        
        Args:
            text (str): Text to split
            header (Optional[str]): Section header
            page_num (int): Page number
            document_name (str): Document name
            
        Returns:
            List[Dict]: List of smaller text segments
        """
        # Split by sentences first
        sentences = re.split(r'[.!?]+', text)
        segments = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            if current_length + len(sentence) > self.max_section_length and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                segments.append({
                    'text': chunk_text,
                    'header': header,
                    'page': page_num,
                    'document': document_name,
                    'length': len(chunk_text)
                })
                current_chunk = [sentence]
                current_length = len(sentence)
            else:
                current_chunk.append(sentence)
                current_length += len(sentence) + 1
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            segments.append({
                'text': chunk_text,
                'header': header,
                'page': page_num,
                'document': document_name,
                'length': len(chunk_text)
            })
        
        return segments


class EmbeddingIndexer:
    """
    A class to create semantic embeddings and FAISS index for document segments.
    
    This class uses sentence-transformers to generate embeddings for text segments
    and creates a FAISS index for efficient similarity search.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding indexer.
        
        Args:
            model_name (str): Name of the sentence transformer model to use
        """
        self.model_name = model_name
        self.model = None
        self.index = None
        self.metadata = []
        self.segmenter = DocumentSegmenter()
        
        # Check if CUDA is available for faster processing
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
    
    def load_model(self):
        """Load the sentence transformer model."""
        if self.model is None:
            logger.info(f"Loading sentence transformer model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name, device=self.device)
            logger.info("Model loaded successfully")
    
    def process_extracted_data(self, extracted_data_path: str) -> List[Dict]:
        """
        Process extracted text data and create segments.
        
        Args:
            extracted_data_path (str): Path to the extracted text JSON file
            
        Returns:
            List[Dict]: List of processed text segments
        """
        logger.info(f"Loading extracted data from: {extracted_data_path}")
        
        with open(extracted_data_path, 'r', encoding='utf-8') as f:
            extracted_data = json.load(f)
        
        all_segments = []
        
        for doc_name, doc_data in extracted_data.get("documents", {}).items():
            logger.info(f"Processing document: {doc_name}")
            
            for page_num, page_data in doc_data.get("pages", {}).items():
                text = page_data.get("text", "")
                if not text:
                    continue
                
                # Segment the text
                segments = self.segmenter.segment_text(text, int(page_num), doc_name)
                all_segments.extend(segments)
                
                logger.info(f"Created {len(segments)} segments from page {page_num}")
        
        logger.info(f"Total segments created: {len(all_segments)}")
        return all_segments
    
    def create_embeddings(self, segments: List[Dict]) -> np.ndarray:
        """
        Create embeddings for text segments.
        
        Args:
            segments (List[Dict]): List of text segments
            
        Returns:
            np.ndarray: Array of embeddings
        """
        if not segments:
            raise ValueError("No segments provided for embedding creation")
        
        self.load_model()
        
        # Extract text from segments
        texts = [segment['text'] for segment in segments]
        
        logger.info(f"Creating embeddings for {len(texts)} segments...")
        
        # Generate embeddings in batches for efficiency
        batch_size = 32
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = self.model.encode(batch_texts, show_progress_bar=True)
            all_embeddings.append(batch_embeddings)
            
            logger.info(f"Processed batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
        
        # Concatenate all embeddings
        embeddings = np.vstack(all_embeddings)
        
        logger.info(f"Created embeddings with shape: {embeddings.shape}")
        return embeddings
    
    def build_faiss_index(self, embeddings: np.ndarray) -> faiss.Index:
        """
        Build a FAISS index for efficient similarity search.
        
        Args:
            embeddings (np.ndarray): Array of embeddings
            
        Returns:
            faiss.Index: FAISS index for similarity search
        """
        logger.info("Building FAISS index...")
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Add embeddings to index
        index.add(embeddings.astype('float32'))
        
        logger.info(f"FAISS index built with {index.ntotal} vectors")
        return index
    
    def build_complete_index(self, extracted_data_path: str, output_dir: str = "indexing"):
        """
        Build the complete embedding index from extracted text data.
        
        Args:
            extracted_data_path (str): Path to extracted text JSON file
            output_dir (str): Directory to save the index and metadata
        """
        logger.info("Starting complete index building process...")
        
        # Process extracted data into segments
        segments = self.process_extracted_data(extracted_data_path)
        
        if not segments:
            logger.error("No segments found to index")
            return
        
        # Create embeddings
        embeddings = self.create_embeddings(segments)
        
        # Build FAISS index
        self.index = self.build_faiss_index(embeddings)
        
        # Store metadata
        self.metadata = segments
        
        # Save index and metadata
        self.save_index(output_dir)
        
        logger.info("Index building completed successfully")
    
    def save_index(self, output_dir: str):
        """
        Save the FAISS index and metadata to disk.
        
        Args:
            output_dir (str): Directory to save the files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        index_path = output_path / "faiss_index.bin"
        faiss.write_index(self.index, str(index_path))
        logger.info(f"FAISS index saved to: {index_path}")
        
        # Save metadata
        metadata_path = output_path / "standards_data.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        logger.info(f"Metadata saved to: {metadata_path}")
        
        # Save index info
        info = {
            "model_name": self.model_name,
            "total_segments": len(self.metadata),
            "embedding_dimension": self.index.d,
            "index_type": "IndexFlatIP",
            "similarity_metric": "cosine"
        }
        
        info_path = output_path / "index_info.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2)
        logger.info(f"Index info saved to: {info_path}")
    
    def load_index(self, index_dir: str = "indexing"):
        """
        Load the FAISS index and metadata from disk.
        
        Args:
            index_dir (str): Directory containing the index files
        """
        index_path = Path(index_dir)
        
        # Load FAISS index
        faiss_index_path = index_path / "faiss_index.bin"
        if faiss_index_path.exists():
            self.index = faiss.read_index(str(faiss_index_path))
            logger.info(f"FAISS index loaded from: {faiss_index_path}")
        else:
            raise FileNotFoundError(f"FAISS index not found at: {faiss_index_path}")
        
        # Load metadata
        metadata_path = index_path / "standards_data.json"
        if metadata_path.exists():
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            logger.info(f"Metadata loaded from: {metadata_path}")
        else:
            raise FileNotFoundError(f"Metadata not found at: {metadata_path}")
        
        # Load model
        self.load_model()
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for similar segments given a query.
        
        Args:
            query (str): Search query
            top_k (int): Number of top results to return
            
        Returns:
            List[Dict]: List of search results with similarity scores
        """
        if self.index is None or not self.metadata:
            raise ValueError("Index not loaded. Call load_index() first.")
        
        # Encode query
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), top_k)
        
        # Format results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.metadata):
                result = self.metadata[idx].copy()
                result['similarity_score'] = float(score)
                result['rank'] = i + 1
                results.append(result)
        
        return results


def main():
    """
    Main function to build the complete index.
    """
    # Check if extracted data exists
    extracted_data_path = "indexing/extracted_text.json"
    if not Path(extracted_data_path).exists():
        logger.error(f"Extracted text data not found at: {extracted_data_path}")
        logger.info("Please run ocr/extract_text.py first to extract text from PDFs")
        return
    
    # Build index
    indexer = EmbeddingIndexer()
    indexer.build_complete_index(extracted_data_path)
    
    logger.info("Index building completed successfully!")


if __name__ == "__main__":
    main()
