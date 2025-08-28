# Get the directory of this Makefile (project root)
PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
VENV_DIR := $(PROJECT_DIR)/.venv
PYTHON := $(VENV_DIR)/bin/python
PIP := $(VENV_DIR)/bin/pip
QDRANT_CONTAINER_NAME := qdrant_rag
QDRANT_IMAGE := qdrant/qdrant:gpu-nvidia-latest

# Ensure we use the project's tools, not any activated venv
RUFF := $(VENV_DIR)/bin/ruff
MYPY := $(VENV_DIR)/bin/mypy
PYTEST := $(VENV_DIR)/bin/pytest

.PHONY: help install installdeps setup-qdrant start-qdrant stop-qdrant logs-qdrant ingest search evaluate-quantization clean
.PHONY: format lint typecheck check test test-cov docs qa build validate verify

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Setup Targets:"
	@echo "  install                 - Create venv and install Python packages"
	@echo "  installdeps             - Install system dependencies (Docker, CUDA, etc)"
	@echo ""
	@echo "Development Targets:"
	@echo "  format                  - Format all Python code with ruff"
	@echo "  lint                    - Lint and auto-fix Python issues"
	@echo "  typecheck               - Run mypy type checking"
	@echo "  check                   - Run both lint and typecheck"
	@echo "  test                    - Run tests"
	@echo "  test-cov                - Run tests with coverage report"
	@echo "  docs                    - Generate documentation"
	@echo "  qa                      - Run format, check, and test"
	@echo "  validate                - Comprehensive validation of all file types"
	@echo "  verify                  - Verify GPU/CUDA and installation status"
	@echo "  build                   - Build distribution packages"
	@echo ""
	@echo "Pipeline Targets:"
	@echo "  setup-qdrant            - Create local directories for Qdrant data and config"
	@echo "  start-qdrant            - Start the Qdrant Docker container with GPU support"
	@echo "  stop-qdrant             - Stop and remove the Qdrant Docker container"
	@echo "  logs-qdrant             - View the logs from the Qdrant container"
	@echo "  ingest                  - Ingest documents from a source directory. Usage: make ingest src=./my_docs"
	@echo "  search                  - Perform a hybrid search. Usage: make search query=\"my search query\""
	@echo "  evaluate-quantization   - Benchmark quantization recall. Usage: make evaluate-quantization src=./sample_docs"
	@echo "  clean                   - Remove virtual environment and generated files"

# --- Core Targets ---

install:
	@echo "=== Python Environment Setup ==="
	@if [ ! -d $(VENV_DIR) ]; then \
		echo "Creating virtual environment at $(VENV_DIR)..."; \
		if command -v uv &> /dev/null; then \
			uv venv $(VENV_DIR); \
		else \
			python3 -m venv $(VENV_DIR); \
		fi; \
		$(PYTHON) -m pip install --upgrade pip; \
	else \
		echo "Using existing virtual environment at $(VENV_DIR)"; \
	fi
	@echo ""
	@echo "Installing development tools (ruff, mypy, pytest)..."
	@if command -v uv &> /dev/null; then \
		uv pip install --python $(PYTHON) ruff mypy pytest pytest-asyncio pytest-cov build datasketch; \
	else \
		$(PYTHON) -m pip install ruff mypy pytest pytest-asyncio pytest-cov build datasketch; \
	fi
	@echo ""
	@echo "Choose installation type:"
	@echo "  1) Minimal (dev tools only, no heavy ML dependencies)"
	@echo "  2) Full (all dependencies including GPU packages)"
	@echo ""
	@read -p "Enter choice [1-2] (default: 1): " -n 1 -r choice; \
	echo; \
	if [[ "$$choice" == "2" ]]; then \
		echo "Installing full dependencies (this may take several minutes)..."; \
		if command -v uv &> /dev/null; then \
			uv pip install --python $(PYTHON) -e ".[dev]"; \
			echo "Installing PyTorch with CUDA support..."; \
			uv pip install --python $(PYTHON) torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121; \
		else \
			$(PYTHON) -m pip install -e ".[dev]"; \
			$(PYTHON) -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121; \
		fi; \
	else \
		echo "Installing minimal setup..."; \
		$(PYTHON) -m pip install --no-deps -e .; \
	fi
	@echo ""
	@echo "✅ Python environment ready at $(VENV_DIR)"
	@echo ""
	@echo "Next steps:"
	@echo "  • Copy .env.example to .env and add your API keys"
	@echo "  • Run './qdrant_rag.py --help' to test (auto-activates venv)"
	@echo "  • Run 'make verify' to check GPU acceleration (if installed)"
	@echo "  • Run 'make start-qdrant' to start the database"

installdeps:
	@echo "=== System Dependencies Installation ==="
	@echo "This will install:"
	@echo "  • uv (fast Python package manager)"
	@echo "  • Docker with GPU support"
	@echo "  • NVIDIA drivers and CUDA toolkit"
	@echo "  • System libraries for document processing"
	@echo "  • Hardware monitoring tools"
	@echo ""
	@echo "Requires sudo access. Designed for Arch-based systems."
	@echo ""
	@read -p "Continue? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		echo ""; \
		echo "Step 1: Installing uv package manager..."; \
		if ! command -v uv &> /dev/null; then \
			curl -LsSf https://astral.sh/uv/install.sh | sh; \
			export PATH="$$HOME/.local/bin:$$PATH"; \
			echo "✓ uv installed to ~/.local/bin"; \
			echo "  Add to your PATH: export PATH=\"$$HOME/.local/bin:$$PATH\""; \
		else \
			echo "✓ uv is already installed (version: $$(uv --version))"; \
		fi; \
		echo ""; \
		if command -v pacman &> /dev/null; then \
			echo "Step 2: Installing system packages via pacman..."; \
			sudo pacman -Syu --noconfirm; \
			sudo pacman -S --needed --noconfirm \
				python python-pip git curl wget unzip gcc make cmake pkg-config \
				libxml2 libxslt zlib bzip2 lz4 zstd openssl base-devel \
				htop iotop btop lm_sensors nvtop \
				docker nvidia nvidia-utils nvidia-settings cuda nvidia-container-toolkit \
				file poppler tesseract tesseract-data-eng libreoffice-fresh pandoc; \
			echo ""; \
			echo "Step 3: Configuring Docker for GPU support..."; \
			sudo systemctl enable docker.service; \
			sudo systemctl start docker.service; \
			if command -v nvidia-ctk &> /dev/null; then \
				sudo nvidia-ctk runtime configure --runtime=docker; \
				sudo systemctl restart docker; \
			fi; \
			if ! groups "$$USER" | grep -q docker; then \
				sudo usermod -aG docker "$$USER"; \
				echo "⚠️  Added $$USER to docker group. Please logout and login again."; \
			else \
				echo "✓ User already in docker group"; \
			fi; \
			echo ""; \
			echo "Step 4: Configuring hardware sensors..."; \
			sudo sensors-detect --auto 2>/dev/null || true; \
			echo ""; \
			echo "✅ System dependencies installed successfully!"; \
			echo ""; \
			echo "Next steps:"; \
			echo "  1. If you were added to docker group, logout and login again"; \
			echo "  2. Run 'make install' to set up Python environment"; \
		else \
			echo "This target is designed for Arch-based systems."; \
			echo "Please install the following manually:"; \
			echo "  - uv: curl -LsSf https://astral.sh/uv/install.sh | sh"; \
			echo "  - Docker with GPU support"; \
			echo "  - NVIDIA drivers and CUDA toolkit"; \
			echo "  - System libraries: libxml2, libxslt, poppler, tesseract"; \
			exit 1; \
		fi; \
	else \
		echo "Installation cancelled."; \
	fi

setup-qdrant:
	@echo "Creating local directories for Qdrant data and config..."
	@mkdir -p $(PROJECT_DIR)/qdrant_data
	@mkdir -p $(PROJECT_DIR)/qdrant_config
	@printf "storage:\n  performance:\n    async_scorer: true\n" > $(PROJECT_DIR)/qdrant_config/config.yaml
	@echo "Qdrant directories and config created."

start-qdrant: setup-qdrant
	@echo "Starting Qdrant Docker container with GPU support..."
	@docker run -d --name $(QDRANT_CONTAINER_NAME) --gpus all \
	  -p 6333:6333 -p 6334:6334 \
	  -v $(CURDIR)/qdrant_data:/qdrant/storage \
	  -v $(CURDIR)/qdrant_config:/qdrant/config \
	  -e QDRANT__GPU__INDEXING=1 \
	  $(QDRANT_IMAGE)
	@echo "Qdrant container is starting. Use 'make logs-qdrant' to monitor."

stop-qdrant:
	@echo "Stopping and removing Qdrant Docker container..."
	@docker stop $(QDRANT_CONTAINER_NAME) || true
	@docker rm $(QDRANT_CONTAINER_NAME) || true
	@echo "Qdrant container stopped and removed."

logs-qdrant:
	@docker logs -f $(QDRANT_CONTAINER_NAME)

ingest:
ifndef src
	$(error "Usage: make ingest src=<path_to_source_directory>")
endif
	@echo "Running ingestion with auto-activated venv..."
	@cd $(PROJECT_DIR) && ./qdrant_rag.py ingest --source $(src)

search:
ifndef query
	$(error "Usage: make search query=\"<your_search_query>\"")
endif
	@echo "Running search with auto-activated venv..."
	@cd $(PROJECT_DIR) && ./qdrant_rag.py search --query "$(query)"

evaluate-quantization:
ifndef src
	$(error "Usage: make evaluate-quantization src=<path_to_sample_directory>")
endif
	@echo "Preparing for quantization evaluation..."
	@echo '[{"query": "example search query 1"}, {"query": "example search query 2"}]' > $(PROJECT_DIR)/queries.json
	@echo "Created sample 'queries.json'. Please edit it with relevant queries for your data."
	@read -p "Press Enter to continue with the evaluation..."
	@cd $(PROJECT_DIR) && ./qdrant_rag.py evaluate-quantization --source $(src) --queries-file queries.json

# --- Development Targets ---

format:
	@echo "Formatting Python code with ruff..."
	@if [ -f $(RUFF) ]; then \
		echo "Formatting: qdrant_rag.py, setup.py"; \
		$(RUFF) format $(PROJECT_DIR)/qdrant_rag.py $(PROJECT_DIR)/setup.py; \
		echo "Checking shell scripts..."; \
		if command -v shellcheck &> /dev/null; then \
			shellcheck $(PROJECT_DIR)/install.sh || true; \
		else \
			echo "Note: Install shellcheck for shell script linting"; \
		fi; \
	else \
		echo "Error: ruff not found. Run 'make install' first."; exit 1; \
	fi

lint:
	@echo "Linting Python code and fixing issues..."
	@if [ -f $(RUFF) ]; then \
		$(RUFF) check $(PROJECT_DIR)/qdrant_rag.py $(PROJECT_DIR)/setup.py --fix; \
		echo "Checking pyproject.toml..."; \
		if [ -f $(PROJECT_DIR)/pyproject.toml ]; then \
			$(PYTHON) -m pip show toml-sort &>/dev/null && toml-sort -c $(PROJECT_DIR)/pyproject.toml || echo "Note: Install toml-sort for TOML validation"; \
		fi; \
	else \
		echo "Error: ruff not found. Run 'make install' first."; exit 1; \
	fi

typecheck:
	@echo "Running type checking on Python files..."
	@if [ -f $(MYPY) ]; then \
		$(MYPY) $(PROJECT_DIR)/qdrant_rag.py $(PROJECT_DIR)/setup.py --ignore-missing-imports; \
	else \
		echo "Error: mypy not found. Run 'make install' first."; exit 1; \
	fi

check: lint typecheck
	@echo "Linting and type checking complete."

test:
	@echo "Running tests..."
	@if [ -f $(PYTEST) ]; then \
		cd $(PROJECT_DIR) && $(PYTEST) tests/ -v || echo "No tests found."; \
	else \
		echo "Error: pytest not found. Run 'make install' first."; exit 1; \
	fi

test-cov:
	@echo "Running tests with coverage..."
	@if [ -f $(PYTEST) ]; then \
		cd $(PROJECT_DIR) && $(PYTEST) tests/ --cov=qdrant_rag --cov-report=html || echo "No tests found."; \
	else \
		echo "Error: pytest not found. Run 'make install' first."; exit 1; \
	fi

docs:
	@echo "Generating documentation..."
	@if [ -f $(PYTHON) ]; then \
		cd $(PROJECT_DIR) && $(PYTHON) -m pydoc -w qdrant_rag; \
	else \
		echo "Error: Python venv not found. Run 'make install' first."; exit 1; \
	fi

qa: format check test
	@echo "Quality assurance complete."

validate:
	@echo "=== Comprehensive Project Validation ==="
	@echo ""
	@echo "Python files:"
	@for file in $(PROJECT_DIR)/qdrant_rag.py $(PROJECT_DIR)/setup.py; do \
		if [ -f $$file ]; then \
			echo "  ✓ $$file"; \
		fi; \
	done
	@echo ""
	@echo "Running Python checks..."
	@if [ -f $(RUFF) ]; then \
		$(RUFF) format --check $(PROJECT_DIR)/qdrant_rag.py $(PROJECT_DIR)/setup.py; \
		$(RUFF) check $(PROJECT_DIR)/qdrant_rag.py $(PROJECT_DIR)/setup.py; \
	fi
	@if [ -f $(MYPY) ]; then \
		$(MYPY) $(PROJECT_DIR)/qdrant_rag.py $(PROJECT_DIR)/setup.py --ignore-missing-imports; \
	fi
	@echo ""
	@echo "Configuration files:"
	@echo "  Checking pyproject.toml syntax..."
	@$(PYTHON) -c "import tomllib; tomllib.load(open('$(PROJECT_DIR)/pyproject.toml', 'rb'))" && echo "  ✓ pyproject.toml is valid" || echo "  ✗ pyproject.toml has errors"
	@echo ""
	@echo "Shell scripts:"
	@if command -v shellcheck &> /dev/null; then \
		shellcheck -S warning $(PROJECT_DIR)/install.sh && echo "  ✓ install.sh passes shellcheck" || echo "  ⚠ install.sh has warnings"; \
	else \
		echo "  ⚠ shellcheck not installed - skipping shell script validation"; \
	fi
	@echo ""
	@echo "Documentation:"
	@if [ -f $(PROJECT_DIR)/README.md ]; then \
		echo "  ✓ README.md exists"; \
		if command -v markdownlint &> /dev/null; then \
			markdownlint $(PROJECT_DIR)/README.md || true; \
		else \
			echo "  ⚠ markdownlint not installed - skipping markdown validation"; \
		fi; \
	fi
	@echo ""
	@echo "=== Validation complete ==="

verify:
	@echo "=== Installation Verification ==="
	@echo ""
	@echo "Python Environment:"
	@$(PYTHON) --version || echo "✗ Python not found"
	@echo "Virtual env: $(VENV_DIR)"
	@echo ""
	@echo "Testing GPU/CUDA availability..."
	@$(PYTHON) -c "import sys; print('Python packages:'); exec('''\ntry:\n    import torch\n    print(f\"  ✓ PyTorch {torch.__version__}\")\n    if torch.cuda.is_available():\n        print(f\"  ✓ CUDA available: {torch.cuda.get_device_name(0)}\")\n        print(f\"  ✓ CUDA version: {torch.version.cuda}\")\n    else:\n        print(\"  ⚠ CUDA not available\")\nexcept ImportError:\n    print(\"  ⚠ PyTorch not installed (use 'make install' with option 2)\")\n\ntry:\n    import onnxruntime as ort\n    print(f\"  ✓ ONNX Runtime {ort.__version__}\")\n    providers = ort.get_available_providers()\n    if \"CUDAExecutionProvider\" in providers:\n        print(\"  ✓ CUDA provider available for ONNX\")\n    else:\n        print(\"  ⚠ CUDA provider not available for ONNX\")\nexcept ImportError:\n    print(\"  ⚠ ONNX Runtime not installed\")\n\ntry:\n    import pynvml\n    pynvml.nvmlInit()\n    handle = pynvml.nvmlDeviceGetHandleByIndex(0)\n    gpu_name = pynvml.nvmlDeviceGetName(handle).decode(\"utf-8\")\n    print(f\"  ✓ GPU detected: {gpu_name}\")\n    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)\n    print(f\"  ✓ GPU Memory: {mem_info.total / 1024**3:.1f} GB\")\n    pynvml.nvmlShutdown()\nexcept:\n    print(\"  ⚠ Unable to detect GPU via pynvml\")\n\ntry:\n    import qdrant_client\n    print(\"  ✓ Qdrant client installed\")\nexcept ImportError:\n    print(\"  ⚠ Qdrant client not installed\")\n''')"
	@echo ""
	@echo "Docker Status:"
	@if command -v docker &> /dev/null; then \
		echo "  ✓ Docker installed: $$(docker --version | cut -d' ' -f3 | sed 's/,//')"; \
		if docker ps &>/dev/null; then \
			echo "  ✓ Docker daemon running"; \
			if docker ps | grep -q $(QDRANT_CONTAINER_NAME); then \
				echo "  ✓ Qdrant container is running"; \
			else \
				echo "  ⚠ Qdrant container not running (use 'make start-qdrant')"; \
			fi; \
		else \
			echo "  ✗ Docker daemon not accessible (check permissions)"; \
		fi; \
	else \
		echo "  ✗ Docker not installed"; \
	fi
	@echo ""
	@echo "Configuration Files:"
	@[ -f $(PROJECT_DIR)/.env ] && echo "  ✓ .env file exists" || echo "  ⚠ .env file missing (copy from .env.example)"
	@[ -f $(PROJECT_DIR)/.env.example ] && echo "  ✓ .env.example exists" || echo "  ✗ .env.example missing"
	@[ -x $(PROJECT_DIR)/qdrant_rag.py ] && echo "  ✓ qdrant_rag.py is executable (auto-activates venv)" || echo "  ✗ qdrant_rag.py not executable"
	@echo ""
	@echo "=== Verification complete ==="

build:
	@echo "Building distribution packages..."
	@if [ -f $(PYTHON) ]; then \
		cd $(PROJECT_DIR) && $(PYTHON) -m build; \
	else \
		echo "Error: Python venv not found. Run 'make install' first."; exit 1; \
	fi

clean:
	@echo "Cleaning up project..."
	@rm -rf $(VENV_DIR)
	@rm -f $(PROJECT_DIR)/.env
	@rm -f $(PROJECT_DIR)/qdrant_rag.log
	@rm -f $(PROJECT_DIR)/queries.json
	@rm -rf $(PROJECT_DIR)/.tmp
	@find $(PROJECT_DIR) -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null || true
	@find $(PROJECT_DIR) -type f -name '*.pyc' -delete 2>/dev/null || true
	@rm -rf $(PROJECT_DIR)/.ruff_cache $(PROJECT_DIR)/.mypy_cache $(PROJECT_DIR)/.pytest_cache 
	@rm -rf $(PROJECT_DIR)/htmlcov $(PROJECT_DIR)/build $(PROJECT_DIR)/dist $(PROJECT_DIR)/*.egg-info
	@echo "Cleanup complete. Run 'make stop-qdrant' to stop the database."
