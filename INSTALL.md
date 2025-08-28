# Installation Guide

This project uses a Makefile-based installation system that separates system dependencies from Python packages.

## Quick Start

### For New Systems (Arch Linux/CachyOS)
```bash
# 1. Install system dependencies (Docker, CUDA, uv, etc.)
make installdeps

# 2. Logout and login if Docker group was added

# 3. Install Python environment
make install
# Choose option 2 for full GPU-accelerated installation

# 4. Configure environment
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY

# 5. Verify installation
make verify
```

### For Existing Development Machines
```bash
# Just set up Python environment
make install
# Choose option 1 for minimal or 2 for full installation

# Configure and verify
cp .env.example .env
make verify
```

## Installation Targets

### `make installdeps`
Installs system-level dependencies (Arch Linux only):
- uv (fast Python package manager)
- Docker with GPU support  
- NVIDIA drivers and CUDA toolkit
- Document processing libraries (poppler, tesseract)
- Hardware monitoring tools

**Note:** Requires sudo access and logout/login for Docker group.

### `make install`
Sets up Python virtual environment with two options:
1. **Minimal** - Development tools only (ruff, mypy, pytest)
2. **Full** - All dependencies including GPU packages (PyTorch, ONNX Runtime)

Uses `uv` if available for faster installation, falls back to pip.

### `make verify`
Checks your installation:
- Python environment status
- GPU/CUDA availability
- Docker configuration
- Qdrant container status
- Configuration files

## Running the Pipeline

The script automatically activates its virtual environment and loads the `.env` file:

```bash
# Just run directly - no activation needed!
./qdrant_rag.py --help

# Or use Python directly
python qdrant_rag.py ingest --source /path/to/docs

# The script will:
# 1. Auto-activate the .venv if not already active
# 2. Auto-load .env file if it exists
# 3. Run with the correct dependencies
```

## Troubleshooting

### Docker Permission Issues
If you get "permission denied" errors with Docker:
```bash
sudo usermod -aG docker $USER
# Then logout and login again
```

### GPU Not Detected
1. Check NVIDIA drivers: `nvidia-smi`
2. Ensure CUDA is installed: `nvcc --version`
3. Reinstall with full dependencies: `make install` (choose option 2)

### Missing uv
If uv is not found:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

### Non-Arch Linux Systems
For Ubuntu/Debian:
- Install Docker, NVIDIA drivers, and CUDA manually
- Install system libraries: `apt-get install libxml2-dev libxslt-dev poppler-utils tesseract-ocr`
- Then run `make install`

## Development Workflow

```bash
# Daily development
source activate.sh
make qa          # Run quality checks
make validate    # Comprehensive validation

# Before commits
make format      # Format code
make check       # Lint and typecheck
make test        # Run tests
```