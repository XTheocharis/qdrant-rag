#!/bin/bash
#
# RAG Pipeline Installation Script
# CachyOS + uv optimized for AMD 9950X3D + RTX 5070 Ti + 64GB RAM
#

set -e # Exit on any error

# --- Configuration ---
VENV_NAME=".venv"

# --- Helper Functions ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

print_banner() {
    echo -e "${BLUE}"
    echo "================================================================="
    echo " High-Performance Hybrid RAG Pipeline Installation"
    echo " CachyOS + uv | Qdrant + Gemini + LlamaIndex + Unstructured"
    echo "================================================================="
    echo -e "${NC}"
}

# --- Prerequisite Checks ---
check_cachyos() {
    if ! command -v pacman &> /dev/null; then
        print_error "This script is designed for CachyOS/Arch Linux only."
        exit 1
    fi
    print_status "CachyOS/Arch Linux detected."
}

check_root() {
    if [ "$(id -u)" -eq 0 ]; then
        print_error "Do not run as root. This script uses sudo where needed."
        exit 1
    fi
    if ! sudo -n true 2>/dev/null; then
        print_warning "This script uses sudo. You may be prompted for your password."
    fi
}

# --- Installation Steps ---
install_system_deps() {
    print_status "Installing all required system dependencies via pacman..."
    sudo pacman -Syu --noconfirm

    print_status "Installing core build tools, Python, and utilities..."
    sudo pacman -S --needed --noconfirm \
        python python-pip git curl wget unzip gcc make cmake pkg-config \
        libxml2 libxslt zlib bzip2 lz4 zstd openssl base-devel

    print_status "Installing hardware monitoring and performance tools..."
    sudo pacman -S --needed --noconfirm htop iotop btop lm_sensors nvtop

    print_status "Installing NVIDIA drivers, CUDA, and container toolkit for RTX 5070 Ti..."
    sudo pacman -S --needed --noconfirm nvidia nvidia-utils nvidia-settings cuda nvidia-container-toolkit

    print_status "Installing container runtime (Docker)..."
    sudo pacman -S --needed --noconfirm docker

    print_status "Installing system libraries for Unstructured document parsing..."
    sudo pacman -S --needed --noconfirm file poppler tesseract tesseract-data-eng libreoffice-fresh pandoc

    print_success "All system dependencies installed."
}

configure_system() {
    print_status "Configuring hardware sensors..."
    sudo sensors-detect --auto || print_warning "sensors-detect non-zero exit, continuing..."

    print_status "Configuring Docker for GPU access..."
    sudo systemctl enable docker.service
    sudo systemctl start docker.service
    sudo nvidia-ctk runtime configure --runtime=docker
    sudo systemctl restart docker

    if ! groups "$USER" | grep -q docker; then
        sudo usermod -aG docker "$USER"
        print_warning "Added $USER to docker group."
        print_error "You MUST now logout and login again for the group change to take effect."
        print_error "After logging back in, please re-run this script to complete the installation."
        exit 1
    fi
    print_success "Docker configured for current user and GPU access."
}

install_uv() {
    print_status "Installing uv (high-performance Python package manager)..."
    if command -v uv &> /dev/null; then
        print_status "uv is already installed. Updating..."
        uv self update
    else
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH" # Add to current session's PATH
    fi

    if ! command -v uv &> /dev/null; then
        print_error "uv installation failed. Please add '$HOME/.local/bin' to your PATH."
        exit 1
    fi
    print_success "uv $(uv --version) is installed."
}

setup_python_env() {
    print_status "Setting up Python virtual environment with uv..."
    if [[ -d "$VENV_NAME" ]]; then
        print_warning "Existing virtual environment found. Removing and recreating..."
        rm -rf "$VENV_NAME"
    fi
    uv venv "$VENV_NAME" --python python3
    print_success "Virtual environment created at './$VENV_NAME'."
}

install_python_packages() {
    print_status "Installing Python packages from pyproject.toml using uv..."
    source "$VENV_NAME/bin/activate"
    # Install project in editable mode with dev dependencies for a complete setup
    uv pip install -e .[dev]
    # Install torch separately to handle the extra index URL correctly
    print_status "Installing PyTorch with CUDA support..."
    uv pip install "torch>=2.2.0" "torchvision" "torchaudio" --index-url https://download.pytorch.org/whl/cu121
    deactivate
    print_success "All Python packages installed successfully."
}

create_project_files() {
    print_status "Creating project configuration files..."

    # .env.example
    cat > .env.example << 'EOF'
# --- REQUIRED ---
# Get your Gemini API Key from Google AI Studio
# The new google-genai SDK uses GOOGLE_API_KEY
GOOGLE_API_KEY="your_gemini_api_key_here"

# --- QDRANT CONFIGURATION ---
QDRANT_URL="http://localhost:6333"
# QDRANT_API_KEY= # Only needed if you configure authentication on your local Qdrant instance

# --- EMBEDDING MODEL ---
# Use the public SDK model name. Defaults to 3072 dimensions.
EMBEDDING_MODEL="gemini-embedding-001"

# --- UNSTRUCTURED PARSING STRATEGY ---
# Use "hi_res" for layout-aware PDF/image parsing (slower), or "fast" for text-based (faster)
UNSTRUCTURED_STRATEGY="hi_res"

# --- RERANKER MODEL ---
# Use "BAAI/bge-reranker-v2-m3" for max quality or "cross-encoder/ms-marco-MiniLM-L-6-v2" for speed
RERANKER_MODEL="BAAI/bge-reranker-v2-m3"
EOF
    print_success "Created .env.example file."

    # activate.sh
    cat > activate.sh << 'EOF'
#!/bin/bash
# CachyOS + uv optimized activation script

# Add uv to PATH if not already there
if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
    export PATH="$HOME/.local/bin:$PATH"
fi

# Activate virtual environment
source .venv/bin/activate

# Load environment variables from .env file if it exists
if [[ -f .env ]]; then
    set -a
    source .env
    set +a
    echo "Environment variables loaded from .env"
fi

echo "High-Performance RAG Pipeline Environment Activated"
echo "Python: $(which python)"
echo "To deactivate, run: deactivate"
EOF
    chmod +x activate.sh
    print_success "Created activation script 'activate.sh'."
}

verify_installation() {
    print_status "Verifying installation and hardware acceleration..."
    source "$VENV_NAME/bin/activate"

    python -c "
import sys
import torch
import pynvml
import onnxruntime as ort
print('--- Python & Core Packages ---')
print(f'Python: {sys.version.split()[0]}')
print(f'PyTorch: {torch.__version__}')
print(f'ONNX Runtime: {ort.__version__}')

print('\n--- GPU Verification ---')
try:
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    gpu_name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
    print(f'SUCCESS: pynvml found GPU: {gpu_name}')
    pynvml.nvmlShutdown()
except Exception as e:
    print(f'WARNING: pynvml check failed: {e}')

if 'CUDAExecutionProvider' in ort.get_available_providers():
    print('SUCCESS: ONNX Runtime has CUDAExecutionProvider available.')
    try:
        from fastembed import SparseTextEmbedding
        model = SparseTextEmbedding('prithivida/Splade_PP_en_v1', providers=['CUDAExecutionProvider'])
        session_providers = model.model.model.get_providers()
        if 'CUDAExecutionProvider' in session_providers:
            print('SUCCESS: FastEmbed SPLADE session is actively using GPU (CUDAExecutionProvider).')
        else:
            print(f'ERROR: FastEmbed SPLADE is NOT using GPU. Active providers: {session_providers}')
    except Exception as e:
        print(f'ERROR: FastEmbed failed to initialize with CUDA provider: {e}')
else:
    print('ERROR: ONNX Runtime CUDA provider not found. GPU acceleration for FastEmbed will fail.')
"
    deactivate
    print_success "Installation verification complete."
}

show_final_instructions() {
    print_success "Installation completed successfully!"
    echo
    echo -e "${BLUE}--- CachyOS + uv Optimized Setup Complete ---${NC}"
    echo
    echo "Next Steps:"
    echo "1.  **Configure API Key:**"
    echo "    cp .env.example .env"
    echo "    nano .env  # Add your GOOGLE_API_KEY"
    echo
    echo "2.  **Start Qdrant Database (GPU-Accelerated):**"
    echo "    make start-qdrant"
    echo
    echo "3.  **Activate Environment:**"
    echo "    source activate.sh"
    echo
    echo "4.  **Run Your Pipeline:**"
    echo "    make ingest src=./path/to/your/docs"
    echo "    make search query=\"your search query\""
    echo "    make evaluate-quantization src=./path/to/sample/docs"
    echo
    print_warning "Remember to start the Qdrant Docker container before running the pipeline!"
}

# --- Main Execution ---
main() {
    print_banner
    check_cachyos
    check_root
    install_system_deps
    configure_system
    install_uv
    create_project_files
    setup_python_env
    install_python_packages
    verify_installation
    show_final_instructions
}

main "$@"
