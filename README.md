# AI Standards Assistant

A semantic search system for standards documents that uses AI embeddings to find relevant sections based on natural language queries.

## 🎯 Project Overview

The AI Standards Assistant automatically:
1. **Understands** the meaning of human comments (e.g., "the noise level in the workshop exceeds the limit")
2. **Searches** through a folder of standards (PDFs with text or scanned images)
3. **Finds** the most relevant section/chapter/page that addresses the issue
4. **Returns** the standard name, chapter title, and optionally extracts the page or snippet

## 🏗️ Architecture

```
.
├── data/
│   └── standards/                # Folder containing input PDFs
├── ocr/
│   └── extract_text.py           # OCR + text extraction
├── indexing/
│   ├── build_index.py            # Creates embeddings + FAISS index
│   └── standards_data.json       # Structured text output
├── api/
│   └── app.py                    # FastAPI app for querying
├── utils/
│   └── pdf_utils.py              # Helper to extract pages/snippets
└── requirements.txt
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd AI_Standards_Assistant

# Install dependencies
pip install -r requirements.txt
```

### 2. Add Your PDFs

Place your standards PDF files in the `data/standards/` folder:

```bash
mkdir -p data/standards
# Copy your PDF files to data/standards/
```

### 3. Extract Text and Build Index

```bash
# Extract text from PDFs (with OCR for image-based PDFs)
python ocr/extract_text.py

# Build the semantic search index
python indexing/build_index.py
```

### 4. Start the API Server

```bash
# Start the FastAPI server
python api/app.py
```

The API will be available at `http://localhost:8000`

## 📚 API Usage

### Interactive Documentation

Visit `http://localhost:8000/docs` for interactive API documentation.

### Search Endpoints

#### POST /search
```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "noise level in workshop exceeds limit",
    "top_k": 5,
    "include_preview": true
  }'
```

#### GET /search
```bash
curl "http://localhost:8000/search?q=noise%20level%20workshop&top_k=3&include_preview=true"
```

### Other Endpoints

- `GET /health` - Check API and index status
- `GET /documents` - List all available documents
- `GET /documents/{name}/info` - Get document metadata
- `GET /documents/{name}/pages/{num}/text` - Get page text
- `GET /documents/{name}/pages/{num}/image` - Get page as image

## 🔧 Configuration

### Environment Variables

Create a `.env` file for configuration:

```env
# Tesseract OCR path (if not in system PATH)
TESSERACT_CMD=/usr/bin/tesseract

# API settings
API_HOST=0.0.0.0
API_PORT=8000

# Model settings
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

### Model Configuration

The system uses `all-MiniLM-L6-v2` by default for embeddings. You can change this in `indexing/build_index.py`:

```python
indexer = EmbeddingIndexer(model_name="your-preferred-model")
```

## 🛠️ Technical Details

### OCR Processing

- **Text-based PDFs**: Direct text extraction using PyMuPDF
- **Image-based PDFs**: OCR using Tesseract with `pdf2image`
- **Hybrid PDFs**: Automatic detection and appropriate processing

### Semantic Search

- **Embeddings**: Sentence transformers (`all-MiniLM-L6-v2`)
- **Index**: FAISS for efficient similarity search
- **Similarity**: Cosine similarity on normalized embeddings

### Text Segmentation

The system intelligently segments documents into meaningful sections:
- Chapter boundaries
- Section headers
- Numbered subsections
- Optimal chunk sizes (200-2000 characters)

## 📊 Performance

- **Indexing**: ~1-2 seconds per page (depending on content)
- **Search**: ~50-100ms per query
- **OCR**: ~2-5 seconds per image page
- **Memory**: ~50-100MB for typical document sets

## 🔍 Example Queries

Try these example queries to test the system:

- "noise level in workshop exceeds limit"
- "safety requirements for electrical equipment"
- "fire prevention measures in industrial facilities"
- "maximum temperature limits for machinery"
- "personal protective equipment requirements"

## 🚀 Advanced Features

### Custom Highlighting

```bash
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "safety requirements",
    "highlight_text": "safety",
    "include_preview": true
  }'
```

### Batch Processing

For large document sets, you can process multiple PDFs:

```python
from ocr.extract_text import PDFTextExtractor

extractor = PDFTextExtractor("path/to/your/pdfs")
extractor.save_extracted_data("output.json")
```

## 🐛 Troubleshooting

### Common Issues

1. **Tesseract not found**
   ```bash
   # Install Tesseract
   # Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
   # macOS: brew install tesseract
   # Ubuntu: sudo apt install tesseract-ocr
   ```

2. **CUDA out of memory**
   - The system automatically falls back to CPU if CUDA is unavailable
   - Reduce batch size in `indexing/build_index.py` if needed

3. **PDF processing errors**
   - Ensure PDFs are not password-protected
   - Check file permissions
   - Verify PDF integrity

### Logging

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🔮 Future Enhancements

- **GPT-4 Integration**: Use GPT-4 for better query understanding
- **BLIP-2 Support**: Image understanding for diagrams and charts
- **Multi-language Support**: OCR and search in multiple languages
- **Web Interface**: Browser-based UI for non-technical users
- **Real-time Updates**: Automatic re-indexing when new PDFs are added

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the API documentation at `/docs`
3. Open an issue on GitHub
