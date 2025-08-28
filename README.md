# Qdrant RAG Pipeline

A self-contained retrieval-augmented generation (RAG) pipeline optimized for AMD 9950X3D and RTX 5070 Ti hardware. Implements hybrid dense/sparse vector search using Qdrant with GPU acceleration for document ingestion and retrieval.

## Architecture

**Single-file design**: All functionality is contained in `qdrant_rag.py` - no external configuration files or complex setup required.

**Hybrid search**: Combines dense embeddings (Google Gemini) with sparse representations (TF-IDF/BM25) for improved retrieval accuracy.

**GPU optimization**: CUDA-accelerated embeddings, ONNX Runtime GPU inference, and optimized batch processing for high-throughput document processing.

**Performance monitoring**: Built-in metrics collection, memory usage tracking, and deduplication using MinHash LSH.

## Requirements

### Hardware
- NVIDIA GPU with CUDA support (tested on RTX 5070 Ti)
- 16GB+ RAM recommended for large document collections
- Docker for Qdrant database

### Software
- Python 3.9+
- Docker with NVIDIA Container Toolkit
- CUDA 12.1+ drivers

## Installation

### Quick Start
```bash
./qdrant_rag.py install all
./qdrant_rag.py qdrant start
```

### Manual Installation
```bash
# System dependencies (Arch Linux)
./qdrant_rag.py install system

# Development tools
./qdrant_rag.py install dev

# Python packages
./qdrant_rag.py install

# Verify installation
./qdrant_rag.py verify
```

### Package Installation
```bash
pip install -e .
```
Creates console script accessible as `pipeline` command.

## Usage

### Document Ingestion
```bash
./qdrant_rag.py ingest /path/to/documents
```

Supported formats: PDF, HTML, Markdown, plain text. Automatic format detection and preprocessing with configurable chunk sizes and overlap.

### Search
```bash
./qdrant_rag.py search "query text"
./qdrant_rag.py search --limit 20 "detailed query"
```

Returns ranked results with relevance scores and source metadata.

### Database Management
```bash
./qdrant_rag.py qdrant start     # Start database
./qdrant_rag.py qdrant stop      # Stop database
./qdrant_rag.py qdrant logs      # View logs
./qdrant_rag.py qdrant erase data  # Clear all data
```

### Performance Analysis
```bash
./qdrant_rag.py benchmark /path/to/test/docs queries.json
```

Evaluates quantization impact on retrieval accuracy using provided test queries.

## Configuration

### Environment Variables
Create `.env` file (copy from `.env.example`):
```bash
GEMINI_API_KEY=your_api_key_here
QDRANT_HOST=localhost
QDRANT_PORT=6333
```

### GPU Configuration
Automatic CUDA detection with fallback to CPU. ONNX Runtime GPU provider used for embedding inference when available.

### Memory Management
- Automatic batch size adjustment based on available GPU memory
- Document chunking with configurable overlap
- MinHash-based deduplication to reduce storage requirements

## Development

### Code Quality
```bash
./qdrant_rag.py format      # Format with ruff
./qdrant_rag.py lint        # Lint and auto-fix
./qdrant_rag.py typecheck   # Type checking with mypy
./qdrant_rag.py test        # Run tests
./qdrant_rag.py qa          # Full quality assurance
```

### Build and Distribution
```bash
./qdrant_rag.py build       # Build wheel/sdist
./qdrant_rag.py validate    # Comprehensive validation
```

### Project Structure
```
qdrant-rag/
├── qdrant_rag.py          # Main application (self-contained)
├── pyproject.toml         # Package configuration
├── .env.example          # Environment template
└── README.md             # Documentation
```

## Implementation Details

### Embedding Strategy
- **Dense**: Google Gemini embedding model via API
- **Sparse**: TF-IDF vectorization with configurable vocabulary
- **Fusion**: RRF (Reciprocal Rank Fusion) for result combination

### Document Processing
- Unstructured library for format detection and parsing
- Semantic chunking with LlamaIndex node parser
- Content cleaning and normalization
- Automatic language detection

### Storage Backend
- Qdrant vector database with GPU-accelerated indexing
- Collections: `documents` (dense), `sparse_documents` (sparse)
- Quantization support for memory optimization
- Configurable distance metrics

### Performance Optimizations
- Batch processing with dynamic sizing
- Connection pooling and request batching
- Memory-mapped file handling for large documents
- Async I/O for concurrent operations

## Benchmarking

The system includes comprehensive benchmarking tools for evaluating different configurations:

```bash
# Create test queries file
echo '[{"query": "machine learning"}, {"query": "neural networks"}]' > queries.json

# Run benchmark
./qdrant_rag.py benchmark ./test_docs queries.json
```

Metrics include retrieval accuracy, query latency, and memory usage across different quantization settings.

## Troubleshooting

### Common Issues

**CUDA not available**: Verify NVIDIA drivers and CUDA toolkit installation. Check `nvidia-smi` output.

**Memory errors**: Reduce batch size or enable quantization. Monitor GPU memory with `nvidia-smi`.

**Docker permission denied**: Add user to docker group: `sudo usermod -aG docker $USER`

**Qdrant connection failed**: Ensure container is running with `./qdrant_rag.py qdrant logs`

### Debug Mode
Set `PYTHONPATH` and run with verbose logging:
```bash
PYTHONPATH=. python -m qdrant_rag --help
```

### Performance Monitoring
Built-in metrics collection tracks:
- Query response times
- Memory usage patterns
- GPU utilization
- Index build performance

## License

This project is developed as a personal tool for document retrieval and analysis. Use at your own discretion.