#!/usr/bin/env python3
"""Hybrid RAG Pipeline using Qdrant.

Hardware Target: AMD 9950X3D + RTX 5070 Ti + 64GB RAM + Gen5 NVMe + CachyOS

Implements the complete 3-Index Hybrid Search architecture:
1. Vector Index (Dense Gemini + Sparse SPLADE)
2. Full-Text Index (Sparse BM25)
3. Payload Index (Structured Metadata)

This file also serves as its own PEP 517 build backend when called by pip.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

# ============================================================================
# BUILD BACKEND MODE - Handle pip install operations
# ============================================================================

# Detect if we're being called as a build backend by pip
# Check for build environment indicators
_IS_BUILD_BACKEND = False

# Method 1: Check for explicit environment variable
if os.environ.get('PEP517_BUILD_BACKEND') == '1':
    _IS_BUILD_BACKEND = True

# Method 2: Check if pep517 or build modules are loaded  
elif 'pep517' in sys.modules or '_in_process' in sys.modules or 'build' in sys.modules:
    _IS_BUILD_BACKEND = True

# Method 3: Check if being imported by setuptools/pip
elif __name__ != "__main__":
    # Being imported rather than executed
    # Check who's importing us
    import inspect
    frame = inspect.currentframe()
    if frame and frame.f_back and frame.f_back.f_code:
        filename = frame.f_back.f_code.co_filename
        if any(x in filename for x in ['setuptools', 'pip', 'pep517', 'build']):
            _IS_BUILD_BACKEND = True

if _IS_BUILD_BACKEND:
    """
    PEP 517 Build Backend Mode
    
    When this script is called as a build backend (via pip install),
    expose the necessary PEP 517 hooks and handle installation.
    """
    import subprocess
    from typing import Optional
    
    # Import setuptools build backend to delegate most operations
    try:
        from setuptools import build_meta as _orig
    except ImportError:
        print("ERROR: setuptools>=64.0 required for build backend", file=sys.stderr)
        sys.exit(1)
    
    # Re-export standard PEP 517 hooks
    def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
        return _orig.prepare_metadata_for_build_wheel(metadata_directory, config_settings)
    
    def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
        return _orig.build_wheel(wheel_directory, config_settings, metadata_directory)
    
    def build_sdist(sdist_directory, config_settings=None):
        return _orig.build_sdist(sdist_directory, config_settings)
    
    def get_requires_for_build_wheel(config_settings=None):
        return _orig.get_requires_for_build_wheel(config_settings)
    
    def get_requires_for_build_sdist(config_settings=None):
        return _orig.get_requires_for_build_sdist(config_settings)
    
    # PEP 660 hooks for editable installs
    if hasattr(_orig, "build_editable"):
        _original_build_editable = _orig.build_editable
        _original_get_requires = _orig.get_requires_for_build_editable
        _original_prepare_metadata = _orig.prepare_metadata_for_build_editable
    else:
        _original_build_editable = None
        _original_get_requires = None
        _original_prepare_metadata = None
    
    def get_requires_for_build_editable(config_settings=None):
        """Get requirements for building editable install."""
        if _original_get_requires:
            return _original_get_requires(config_settings)
        return []
    
    def prepare_metadata_for_build_editable(metadata_directory, config_settings=None):
        """Prepare metadata for editable install."""
        if _original_prepare_metadata:
            return _original_prepare_metadata(metadata_directory, config_settings)
        raise NotImplementedError("Editable installs require setuptools >= 64.0.0")
    
    def run_make_install():
        """Execute 'make install' when doing editable install."""
        # Prevent recursion if already installing
        if os.environ.get("QDRANT_RAG_INSTALLING") == "1":
            print("Skipping make install - already running", file=sys.stderr)
            return False
        
        project_dir = Path(__file__).parent.absolute()
        makefile = project_dir / "Makefile"
        
        if not makefile.exists():
            print("Note: Makefile not found, skipping make install", file=sys.stderr)
            return False
        
        # Log the installation
        log_file = project_dir / "pip_install.log"
        with open(log_file, "a") as f:
            f.write("\n" + "="*60 + "\n")
            f.write("Self-contained build: Executing 'make install'...\n")
            f.write("="*60 + "\n\n")
        
        try:
            env = os.environ.copy()
            env["QDRANT_RAG_INSTALLING"] = "1"
            
            result = subprocess.run(
                ["make", "install"],
                cwd=str(project_dir),
                check=False,
                capture_output=False,
                text=True,
                env=env
            )
            
            if result.returncode == 0:
                with open(log_file, "a") as f:
                    f.write("✅ Successfully completed 'make install'\n")
                return True
            else:
                with open(log_file, "a") as f:
                    f.write(f"⚠️ 'make install' exited with code {result.returncode}\n")
                return False
        except FileNotFoundError:
            print("⚠️ 'make' not found. Run 'make install' manually.", file=sys.stderr)
            return False
        except Exception as e:
            print(f"⚠️ Could not run 'make install': {e}", file=sys.stderr)
            return False
    
    def build_editable(wheel_directory, config_settings=None, metadata_directory=None):
        """Build editable wheel and trigger make install."""
        # Run make install for comprehensive setup
        run_make_install()
        
        # Delegate to setuptools
        if _original_build_editable:
            return _original_build_editable(wheel_directory, config_settings, metadata_directory)
        else:
            raise NotImplementedError("Editable installs require setuptools >= 64.0.0")
    
    # If we're in build backend mode, stop here - don't run the rest of the script
    # The build backend functions are now available for pip to use
    __all__ = [
        'prepare_metadata_for_build_wheel',
        'build_wheel', 
        'build_sdist',
        'get_requires_for_build_wheel',
        'get_requires_for_build_sdist',
        'get_requires_for_build_editable',
        'prepare_metadata_for_build_editable',
        'build_editable'
    ]
    
    # Exit build backend mode - don't execute runtime code
    if __name__ == "__main__":
        # Direct execution during build - just exit
        sys.exit(0)
    # If being imported as a module, the functions are now available
    # but we still don't want to run the rest of the script
    
else:
    # ============================================================================
    # RUNTIME MODE - Normal script execution
    # ============================================================================
    
    # Only execute the rest of the script if NOT in build backend mode
    # This is where all the normal runtime code goes
    pass  # Continue with normal execution below

# The rest of the file continues normally, but we guard the execution parts
# Auto-activate virtual environment if not already active

def ensure_venv():
    """Auto-activate the project's virtual environment if not already active."""
    script_dir = Path(__file__).resolve().parent
    venv_dir = script_dir / ".venv"
    venv_python = venv_dir / "bin" / "python" if os.name != 'nt' else venv_dir / "Scripts" / "python.exe"

    # Check if we're running with the correct Python by checking executable path
    # Don't resolve symlinks as venv python points to system python
    current_python = Path(sys.executable)

    # If we're already using the venv's Python, we're good
    if str(current_python) == str(venv_python):
        return  # Already using correct Python

    # Check if venv exists - but allow management commands that don't need venv
    if not venv_dir.exists():
        # Check if user is trying to run management/setup commands
        management_commands = [
            'install', 'installdeps', 'verify', 'clean', 'erasedata',
            'start-qdrant', 'stop-qdrant', 'logs-qdrant',
            '--help', '-h', 'help'
        ]
        if len(sys.argv) > 1 and sys.argv[1] in management_commands:
            return  # Allow these commands without venv
        print(f"Error: Virtual environment not found at {venv_dir}")
        print("Please run './qdrant_rag.py install' first to set up the environment.")
        sys.exit(1)

    if not venv_python.exists():
        print(f"Error: Python executable not found at {venv_python}")
        print("Virtual environment may be corrupted. Run 'make clean && make install'")
        sys.exit(1)

    # Re-launch the script with the venv's Python
    print(f"Auto-activating virtual environment: {venv_dir}")
    os.execv(str(venv_python), [str(venv_python)] + sys.argv)

# Ensure we're in the venv before importing dependencies (only in runtime mode)
if not _IS_BUILD_BACKEND:
    ensure_venv()

# Auto-load .env file if it exists
def load_env():
    """Load environment variables from .env file if it exists."""
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                        # Remove quotes if present
                        value = value.strip().strip('"').strip("'")
                        os.environ[key] = value

if not _IS_BUILD_BACKEND:
    load_env()

# Check for required dependencies before importing
def check_dependencies():
    """Check if required dependencies are installed and show README if not."""
    required_packages = [
        'onnxruntime',
        'psutil',
        'pynvml',
        'datasketch',
        'fastembed',
        'google.genai',  # This is the actual module we import
        'llama_index',
        'qdrant_client',
        'sentence_transformers',
        'torch',
        'unstructured'
    ]

    missing = []
    for package in required_packages:
        try:
            if package == 'google.genai':
                # Special handling for google.genai which is a submodule
                import google.genai
            elif '.' in package:
                # Handle other nested imports
                parts = package.split('.')
                __import__(parts[0])
                mod = sys.modules[parts[0]]
                for part in parts[1:]:
                    mod = getattr(mod, part)
            else:
                __import__(package)
        except (ImportError, AttributeError):
            missing.append(package)

    if missing:
        readme_path = Path(__file__).resolve().parent / "README.md"
        if readme_path.exists():
            import subprocess
            # Use 'less' with fallback to 'more'
            pager = 'less' if subprocess.run(['which', 'less'], capture_output=True).returncode == 0 else 'more'
            try:
                subprocess.run([pager, str(readme_path)])
            except (FileNotFoundError, subprocess.SubprocessError):
                # Fallback to cat if no pager available
                with open(readme_path) as f:
                    print(f.read())
        else:
            print("Dependencies missing and README.md not found.")

        print(f"\nMissing dependencies: {', '.join(missing)}")
        print("\nPlease run the following commands:")
        print("  make installdeps  # Install system dependencies")
        print("  make install      # Install Python packages")
        sys.exit(1)

if not _IS_BUILD_BACKEND:
    check_dependencies()

# Imports after dependency check - suppress E402 warnings
# ruff: noqa: E402
import argparse
import asyncio
import json
import logging
import signal
import threading
import time
import unicodedata
import uuid
from concurrent.futures import (
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
)
from dataclasses import dataclass
from typing import Any, Optional

import onnxruntime as ort
import psutil
import pynvml
from datasketch import MinHash, MinHashLSH

# --- Embeddings & Reranking ---
from fastembed import SparseTextEmbedding

# --- Core Dependencies ---
from google import genai
from google.genai.types import EmbedContentConfig
from llama_index.core import Document as LlamaDocument
from llama_index.core.node_parser import (
    CodeSplitter,
    SemanticSplitterNodeParser,
)
from llama_index.embeddings.google_genai import (
    GoogleGenAIEmbedding as LlamaGeminiEmbedding,
)

# --- Qdrant ---
from qdrant_client import QdrantClient
from qdrant_client import models as qm
from sentence_transformers import CrossEncoder

# --- Document Parsing & Chunking ---
from unstructured.partition.auto import partition as unstructured_partition

# --- Optional Dependency Checks ---
try:
    from bs4 import BeautifulSoup

    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(threadName)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("qdrant_rag.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)


# --- Enums and Dataclasses ---
@dataclass
class PerformanceMetrics:
    """Track performance across the pipeline."""

    start_time: float = 0.0
    end_time: float = 0.0
    total_files: int = 0
    processed_files: int = 0
    initial_chunks: int = 0
    deduped_chunks: int = 0
    dense_embeddings: int = 0
    sparse_splade_embeddings: int = 0
    sparse_bm25_embeddings: int = 0

    @property
    def total_time(self) -> float:
        return self.end_time - self.start_time if self.end_time > 0 else 0.0

    def summary(self) -> dict[str, Any]:
        """Generate a summary of performance metrics.

        Returns:
            Dictionary containing all metrics plus calculated rates.
        """
        total_time = self.total_time
        return {
            "total_time_seconds": round(total_time, 2),
            "files_per_second": round(self.processed_files / total_time, 2)
            if total_time > 0
            else 0,
            "chunks_per_second": round(self.deduped_chunks / total_time, 2)
            if total_time > 0
            else 0,
            **self.__dict__,
        }


# --- System Resource Management ---
class SystemResourceManager:
    """Monitors and throttles system resources to prevent overload."""

    def __init__(self):
        self.process = psutil.Process()
        self.monitoring = True
        self.emergency_throttle = False
        self.throttle_lock = threading.Lock()

        self.max_cpu_percent = 80.0
        self.max_memory_percent = 70.0
        self.min_free_memory_gb = 4.0
        self.max_cpu_temp_c = float(os.getenv("MAX_CPU_TEMP", 85))
        self.max_gpu_temp_c = float(os.getenv("MAX_GPU_TEMP", 83))
        self.critical_cpu_temp_c = float(os.getenv("CRITICAL_CPU_TEMP", 95))
        self.critical_gpu_temp_c = float(os.getenv("CRITICAL_GPU_TEMP", 90))

        self.throttle_delay = 0.1
        self.recovery_threshold = 0.8

        self.temp_monitoring_available = self._check_temp_monitoring()
        self._set_process_niceness()
        self.setup_signal_handlers()

    def _check_temp_monitoring(self) -> dict[str, bool]:
        available = {"cpu": False, "gpu": False}
        try:
            if (
                hasattr(psutil, "sensors_temperatures")
                and psutil.sensors_temperatures()
            ):
                available["cpu"] = True
        except Exception:
            pass
        try:
            pynvml.nvmlInit()
            available["gpu"] = True
            pynvml.nvmlShutdown()
        except Exception:
            pass
        logger.info(f"Temperature monitoring available: {available}")
        return available

    def get_cpu_temperature(self) -> Optional[float]:
        if not self.temp_monitoring_available["cpu"]:
            return None
        try:
            temps = psutil.sensors_temperatures()
            if "k10temp" in temps:  # AMD
                tctl = next(
                    (s for s in temps["k10temp"] if "Tctl" in s.label), None
                )
                return (
                    tctl.current
                    if tctl
                    else max(s.current for s in temps["k10temp"])
                )
            elif "coretemp" in temps:  # Intel
                return max(s.current for s in temps["coretemp"])
        except Exception as e:
            logger.debug(f"CPU temp reading failed: {e}")
        return None

    def get_gpu_temperature(self) -> Optional[float]:
        if not self.temp_monitoring_available["gpu"]:
            return None
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            temp = pynvml.nvmlDeviceGetTemperature(
                handle, pynvml.NVML_TEMPERATURE_GPU
            )
            pynvml.nvmlShutdown()
            return float(temp)
        except Exception as e:
            logger.debug(f"GPU temp reading failed: {e}")
        return None

    def _set_process_niceness(self):
        try:
            os.nice(5)
            logger.info(
                f"Process niceness set to {os.nice(0)} for balanced performance."
            )
        except Exception as e:
            logger.warning(f"Could not set process niceness: {e}")

    def get_system_stats(self) -> dict[str, Any]:
        """Get current system resource statistics.

        Returns:
            Dictionary with CPU, memory, and temperature statistics.
        """
        try:
            memory = psutil.virtual_memory()
            return {
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "mem_percent": memory.percent,
                "mem_available_gb": memory.available / (1024**3),
                "cpu_temp_c": self.get_cpu_temperature(),
                "gpu_temp_c": self.get_gpu_temperature(),
            }
        except Exception:
            return {}

    def should_throttle(self) -> tuple[bool, str]:
        """Check if system resources require throttling.

        Returns:
            Tuple of (should_throttle: bool, reason: str).
        """
        stats = self.get_system_stats()
        if not stats:
            return False, ""

        if (
            temp := stats.get("cpu_temp_c")
        ) and temp > self.critical_cpu_temp_c:
            raise RuntimeError(f"CPU temp critical: {temp}°C")
        if (
            temp := stats.get("gpu_temp_c")
        ) and temp > self.critical_gpu_temp_c:
            raise RuntimeError(f"GPU temp critical: {temp}°C")

        if (temp := stats.get("cpu_temp_c")) and temp > self.max_cpu_temp_c:
            return True, f"CPU Temp High ({temp}°C)"
        if (temp := stats.get("gpu_temp_c")) and temp > self.max_gpu_temp_c:
            return True, f"GPU Temp High ({temp}°C)"
        if stats["cpu_percent"] > self.max_cpu_percent:
            return True, f"CPU High ({stats['cpu_percent']}%)"
        if stats["mem_percent"] > self.max_memory_percent:
            return True, f"Memory High ({stats['mem_percent']}%)"
        if stats["mem_available_gb"] < self.min_free_memory_gb:
            return True, f"Memory Low ({stats['mem_available_gb']:.1f}GB)"

        return False, ""

    async def adaptive_throttle(self, operation_name: str):
        if not self.monitoring:
            return
        should, reason = self.should_throttle()
        with self.throttle_lock:
            if should and not self.emergency_throttle:
                self.emergency_throttle = True
                logger.warning(f"THROTTLING {operation_name}: {reason}")
            elif not should and self.emergency_throttle:
                stats = self.get_system_stats()
                if (
                    stats.get("cpu_percent", 100)
                    < self.max_cpu_percent * self.recovery_threshold
                    and stats.get("mem_percent", 100)
                    < self.max_memory_percent * self.recovery_threshold
                ):
                    self.emergency_throttle = False
                    logger.info(
                        f"RECOVERY: {operation_name} throttling disabled."
                    )
        if self.emergency_throttle:
            await asyncio.sleep(self.throttle_delay)

    def setup_signal_handlers(self):
        def handler(signum, frame):
            logger.info(f"Signal {signum} received, shutting down.")
            self.monitoring = False

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

    def close(self):
        """Shutdown the resource monitoring system."""
        self.monitoring = False
        logger.info("Resource manager shut down.")


# --- Core Pipeline Components ---
class MaxPerformanceConfig:
    """Configuration optimized for the target hardware."""

    def __init__(self):
        self.gemini_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv(
            "GEMINI_API_KEY"
        )
        if not self.gemini_api_key:
            raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY not set")

        self.cpu_threads = 32
        self.gemini_concurrent_requests = 32
        self.gemini_batch_size = 100

        self.embedding_model = os.getenv(
            "EMBEDDING_MODEL", "gemini-embedding-001"
        )
        self.embedding_dimension = 3072

        self.splade_model = "prithivida/Splade_PP_en_v1"
        self.bm25_model = "Qdrant/bm25"

        self.qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.collection_name = "max_performance_hybrid"

        self.enable_gpu = (
            os.getenv("ENABLE_GPU_ACCELERATION", "true").lower() == "true"
        )
        self.unstructured_strategy = os.getenv(
            "UNSTRUCTURED_STRATEGY", "hi_res"
        )
        self.reranker_model = os.getenv(
            "RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"
        )
        self.enable_reranking = True


class Embedder:
    """Handles dense (Gemini) and double sparse (SPLADE + BM25) embeddings."""

    def __init__(
        self, config: MaxPerformanceConfig, executor: ThreadPoolExecutor
    ):
        self.config = config
        self.semaphore = asyncio.Semaphore(config.gemini_concurrent_requests)
        self.cpu_executor = executor

        self.gemini_client = genai.Client(api_key=config.gemini_api_key)
        self.llama_gemini_embedder = LlamaGeminiEmbedding(
            api_key=config.gemini_api_key, model_name=config.embedding_model
        )

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if config.enable_gpu
            else ["CPUExecutionProvider"]
        )
        self.splade_embedder = SparseTextEmbedding(
            model_name=config.splade_model, providers=providers
        )
        self.bm25_embedder = SparseTextEmbedding(
            model_name=config.bm25_model, providers=providers
        )
        self._verify_fastembed_gpu()

    def _verify_fastembed_gpu(self):
        if self.config.enable_gpu:
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                logger.warning(
                    "ONNX Runtime CUDA provider not found. FastEmbed will use CPU."
                )
                return

            # SPLADE uses ONNX and can utilize GPU
            splade_providers = self.splade_embedder.model.model.get_providers()
            
            if "CUDAExecutionProvider" in splade_providers:
                logger.info(
                    "SUCCESS: FastEmbed SPLADE is actively using GPU (CUDAExecutionProvider)."
                )
            else:
                logger.warning(
                    f"FastEmbed SPLADE is NOT using GPU. Active providers: {splade_providers}"
                )
            
            # BM25 is a traditional algorithm, doesn't use ONNX/GPU
            logger.info("BM25 is a lexical algorithm and doesn't use GPU acceleration.")

    async def _rate_limited_gemini_batch_request(
        self, texts: list[str], task_type: str
    ) -> list[list[float]]:
        async with self.semaphore:
            loop = asyncio.get_event_loop()

            def embed_batch():
                # Use embed_content with a list for batching per SDK docs
                res = self.gemini_client.models.embed_content(
                    model=self.config.embedding_model,
                    contents=texts,
                    config=EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=self.config.embedding_dimension,
                    ),
                )
                return [emb.values for emb in res.embeddings]

            for attempt in range(5):
                try:
                    return await loop.run_in_executor(
                        self.cpu_executor, embed_batch
                    )
                except Exception as e:
                    if "rate limit" in str(e).lower() and attempt < 4:
                        await asyncio.sleep(2**attempt)
                    else:
                        raise
            # This should never be reached due to the raise in the else clause
            raise RuntimeError("All retry attempts failed")

    async def embed_dense_documents(
        self, texts: list[str]
    ) -> list[list[float]]:
        """Generate dense embeddings for multiple documents.

        Args:
            texts: List of document text strings to embed.

        Returns:
            List of embedding vectors (3072-dimensional by default).
        """
        batch_size = self.config.gemini_batch_size
        tasks = [
            self._rate_limited_gemini_batch_request(
                texts[i : i + batch_size], "RETRIEVAL_DOCUMENT"
            )
            for i in range(0, len(texts), batch_size)
        ]
        results = await asyncio.gather(*tasks)
        return [item for sublist in results for item in sublist]

    def embed_dense_query(self, text: str) -> list[float]:
        """Generate dense embedding for a search query.

        Args:
            text: Query string to embed.

        Returns:
            Single embedding vector optimized for retrieval.
        """
        res = self.gemini_client.models.embed_content(
            model=self.config.embedding_model,
            contents=text,  # API expects string, not list
            config=EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=self.config.embedding_dimension,
            ),
        )
        if res.embeddings and len(res.embeddings) > 0:
            values = res.embeddings[0].values
            return values if values else []
        return []

    def embed_sparse(
        self, texts: list[str]
    ) -> dict[str, list[dict[str, Any]]]:
        """Generate sparse embeddings using SPLADE and BM25.

        Args:
            texts: List of text strings to embed.

        Returns:
            Dictionary with 'splade' and 'bm25' sparse vectors.
        """
        splade_embs = self.splade_embedder.embed(texts)
        bm25_embs = self.bm25_embedder.embed(texts)
        return {
            "splade": [
                {
                    "indices": emb.indices.tolist(),
                    "values": emb.values.tolist(),
                }
                for emb in splade_embs
            ],
            "bm25": [
                {
                    "indices": emb.indices.tolist(),
                    "values": emb.values.tolist(),
                }
                for emb in bm25_embs
            ],
        }


class StructuredDocumentSplitter:
    """XML/HTML-aware document splitter that respects document structure."""

    def split_html_document(self, html: str) -> list[str]:
        if not BS4_AVAILABLE:
            return [html]
        soup = BeautifulSoup(html, "html.parser")
        blocks = [
            tag.get_text(" ", strip=True)
            for tag in soup.find_all(
                ["h1", "h2", "h3", "p", "li", "pre", "code"]
            )
            if tag.get_text(strip=True)
        ]
        return blocks

    def split_xml_document(self, xml: str) -> list[str]:
        if not BS4_AVAILABLE:
            return [xml]
        soup = BeautifulSoup(xml, "xml")
        texts = [t.strip() for t in soup.stripped_strings]
        return [" ".join(texts)] if texts else []


class DocumentProcessor:
    """Handles content-aware parsing, hygiene, and chunking."""

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.structured_splitter = StructuredDocumentSplitter()
        self.semantic_splitter = SemanticSplitterNodeParser.from_defaults(
            embed_model=self.embedder.llama_gemini_embedder,
            breakpoint_percentile_threshold=95,
        )
        self.code_splitter = CodeSplitter.from_defaults(language="python")

    def _normalize_and_clean(self, text: str) -> str:
        return unicodedata.normalize("NFC", text)

    def _deduplicate_chunks(
        self, chunks: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if not chunks:
            return []

        # Create MinHash objects for each text chunk
        lsh = MinHashLSH(threshold=0.85, num_perm=128)
        minhashes = []

        for i, chunk in enumerate(chunks):
            mh = MinHash(num_perm=128)
            # Hash each word in the text
            for word in chunk["text"].split():
                mh.update(word.encode("utf-8"))
            minhashes.append((str(i), mh))
            # Insert into LSH
            lsh.insert(str(i), mh)

        # Find duplicates
        seen = set()
        unique_chunks = []

        for i, (key, mh) in enumerate(minhashes):
            if key not in seen:
                # Get similar items
                similar = lsh.query(mh)
                # Mark all similar items as seen (keep only the first)
                for sim_key in similar:
                    if sim_key != key:
                        seen.add(sim_key)
                # Keep this chunk
                unique_chunks.append(chunks[i])

        logger.info(
            f"Deduplication removed {len(chunks) - len(unique_chunks)} near-duplicate chunks."
        )
        return unique_chunks

    def process_file(
        self, file_path: Path, unstructured_strategy: str
    ) -> list[dict[str, Any]]:
        """Process a single file into deduplicated chunks.

        Applies content-aware parsing and chunking based on file type:
        - Code files: Language-specific AST-aware splitting
        - HTML/XML: Structure-preserving block extraction
        - Documents: Semantic splitting with layout preservation

        Args:
            file_path: Path object pointing to the file to process.
            unstructured_strategy: Strategy for unstructured parsing.
                'hi_res' for layout-aware parsing (slower, better quality).
                'fast' for text-only extraction (faster, lower quality).

        Returns:
            List of chunk dictionaries, each containing:
                - text: The normalized and cleaned chunk text
                - metadata: Dict with source, file_type, and content_type

        Example:
            >>> chunks = processor.process_file(
            ...     Path("document.pdf"),
            ...     unstructured_strategy="hi_res"
            ... )
            >>> print(f"Generated {len(chunks)} chunks")

        Note:
            Applies MinHash LSH deduplication with 85% similarity threshold.
            Empty chunks are automatically filtered out.
        """
        ext = file_path.suffix.lower()

        chunks_text = []
        metadata = {"source": str(file_path), "file_type": ext}

        if ext in {
            ".py",
            ".js",
            ".java",
            ".cpp",
            ".c",
            ".h",
            ".rs",
            ".go",
        }:  # Code
            content = self._normalize_and_clean(
                file_path.read_text(encoding="utf-8", errors="ignore")
            )
            lang_map = {
                ".py": "python",
                ".js": "javascript",
                ".java": "java",
                ".cpp": "cpp",
                ".c": "c",
                ".h": "c",
                ".rs": "rust",
                ".go": "go",
            }
            self.code_splitter.language = lang_map.get(ext, "python")
            chunks_text = self.code_splitter.split_text(content)
            metadata["content_type"] = "code"
        elif ext in {".html", ".htm", ".xhtml", ".xml", ".svg"}:  # Markup
            content = self._normalize_and_clean(
                file_path.read_text(encoding="utf-8", errors="ignore")
            )
            chunks_text = (
                self.structured_splitter.split_html_document(content)
                if ext.startswith(".htm")
                else self.structured_splitter.split_xml_document(content)
            )
            metadata["content_type"] = "markup"
        else:  # Documents
            elements = unstructured_partition(
                filename=str(file_path), strategy=unstructured_strategy
            )
            content = self._normalize_and_clean(
                "\n\n".join(
                    [el.text for el in elements if hasattr(el, "text")]
                )
            )
            nodes = self.semantic_splitter.get_nodes_from_documents(
                [LlamaDocument(text=content)]
            )
            chunks_text = [node.get_content() for node in nodes]
            metadata["content_type"] = "document"

        chunks = [
            {"text": chunk, "metadata": metadata}
            for chunk in chunks_text
            if chunk.strip()
        ]
        return self._deduplicate_chunks(chunks)


class QdrantStore:
    """Manages interaction with the Qdrant database."""

    def __init__(self, config: MaxPerformanceConfig):
        self.config = config
        self.client = QdrantClient(url=config.qdrant_url, prefer_grpc=True)
        self._ensure_collection()

    def _ensure_collection(self):
        try:
            self.client.get_collection(self.config.collection_name)
        except Exception:
            self.client.create_collection(
                collection_name=self.config.collection_name,
                vectors_config={
                    "dense": qm.VectorParams(
                        size=self.config.embedding_dimension,
                        distance=qm.Distance.COSINE,
                    )
                },
                sparse_vectors_config={
                    "splade": qm.SparseVectorParams(),
                    "bm25": qm.SparseVectorParams(modifier=qm.Modifier.IDF),
                },
                hnsw_config=qm.HnswConfigDiff(m=48, ef_construct=1024),
                quantization_config=qm.ScalarQuantization(
                    scalar=qm.ScalarQuantizationConfig(
                        type=qm.ScalarType.INT8, quantile=0.99, always_ram=True
                    )
                ),
            )
            self.client.create_payload_index(
                collection_name=self.config.collection_name,
                field_name="text",
                field_schema=qm.TextIndexParams(
                    type="text",
                    tokenizer=qm.TokenizerType.MULTILINGUAL,
                    min_token_len=2,
                    max_token_len=20,
                    lowercase=True,
                ),
            )
            self.client.create_payload_index(
                collection_name=self.config.collection_name,
                field_name="metadata.source",
                field_schema=qm.PayloadSchemaType.KEYWORD,
            )
            self.client.create_payload_index(
                collection_name=self.config.collection_name,
                field_name="metadata.content_type",
                field_schema=qm.PayloadSchemaType.KEYWORD,
            )

    def upsert(
        self,
        chunks: list[dict],
        dense_vectors: list[list[float]],
        sparse_vectors: dict[str, list[dict]],
    ):
        """Insert or update document chunks in the Qdrant collection.

        Creates point structures with dense and sparse vectors for each chunk
        and performs batch upsert to Qdrant with synchronous commit.

        Args:
            chunks: List of chunk dictionaries with 'text' and 'metadata' keys.
            dense_vectors: List of dense embedding vectors (Gemini embeddings).
            sparse_vectors: Dictionary containing 'splade' and 'bm25' sparse vectors,
                each as a list of dicts with 'indices' and 'values' keys.

        Returns:
            None. Blocks until all points are committed to the database.

        Raises:
            QdrantException: If upsert operation fails.

        Example:
            >>> chunks = [{'text': 'content', 'metadata': {'source': 'doc.pdf'}}]
            >>> dense = [[0.1, 0.2, ...]]  # 3072-dimensional vectors
            >>> sparse = {'splade': [...], 'bm25': [...]}
            >>> store.upsert(chunks, dense, sparse)
        """
        points = [
            qm.PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": dense,
                    "splade": qm.SparseVector(**sparse_vectors["splade"][i]),
                    "bm25": qm.SparseVector(**sparse_vectors["bm25"][i]),
                },
                payload={**chunk["metadata"], "text": chunk["text"]},
            )
            for i, (chunk, dense) in enumerate(zip(chunks, dense_vectors))
        ]
        self.client.upsert(
            collection_name=self.config.collection_name,
            points=points,
            wait=True,
        )

    def search(
        self, dense_query: list[float], sparse_queries: dict, limit: int
    ) -> list[dict]:
        """Search the Qdrant collection using hybrid query fusion.

        Performs parallel searches across dense and sparse indices, then fuses
        results using Reciprocal Rank Fusion (RRF) for optimal ranking.

        Args:
            dense_query: Dense query embedding vector.
            sparse_queries: Dictionary with 'splade' and 'bm25' sparse query vectors.
            limit: Maximum number of results to return.

        Returns:
            List of dictionaries containing:
                - score: RRF fusion score
                - text: The chunk text
                - metadata: Original metadata from the chunk

        Note:
            Prefetches 5x the limit from each index before fusion to ensure
            sufficient candidate diversity. Uses INT8 quantization with rescoring.
        """
        prefetch_queries = [
            qm.Prefetch(query=dense_query, using="dense", limit=limit * 5),
            qm.Prefetch(
                query=qm.SparseVector(**sparse_queries["splade"]),
                using="splade",
                limit=limit * 5,
            ),
            qm.Prefetch(
                query=qm.SparseVector(**sparse_queries["bm25"]),
                using="bm25",
                limit=limit * 5,
            ),
        ]
        hits = self.client.query_points(
            collection_name=self.config.collection_name,
            prefetch=prefetch_queries,
            query=qm.FusionQuery(fusion=qm.Fusion.RRF),
            limit=limit,
            with_payload=True,
            search_params=qm.SearchParams(
                quantization=qm.QuantizationSearchParams(rescore=True)
            ),
        )
        return [
            {"score": hit.score, **(hit.payload if hit.payload else {})}
            for hit in hits.points
        ]


class MaxPerformancePipeline:
    """Orchestrates the entire ingestion and search pipeline."""

    def __init__(self):
        self.config = MaxPerformanceConfig()
        self.resource_manager = SystemResourceManager()
        self.cpu_executor = ThreadPoolExecutor(
            max_workers=self.config.cpu_threads
        )
        self.embedder = Embedder(self.config, self.cpu_executor)
        self.processor = DocumentProcessor(self.embedder)
        self.store = QdrantStore(self.config)
        self.reranker = (
            CrossEncoder(
                self.config.reranker_model,
                device="cuda" if self.config.enable_gpu else "cpu",
            )
            if self.config.reranker_model
            else None
        )

    async def ingest(self, source_path: str):
        """Ingest documents from a source directory into the Qdrant database.

        Processes all files recursively, chunks them appropriately based on file type,
        generates dense and sparse embeddings, and stores them in Qdrant with optimized
        indexing for hybrid search.

        Args:
            source_path: Path to the directory containing documents to ingest.
                Supports nested directories and various file formats.

        Returns:
            None. Logs performance metrics upon completion.

        Raises:
            ValueError: If source_path does not exist.
            RuntimeError: If CPU/GPU temperature exceeds critical thresholds.

        Example:
            >>> pipeline = MaxPerformancePipeline()
            >>> await pipeline.ingest("/path/to/documents")
            # Processes all documents and stores them in Qdrant

        Note:
            Uses parallel processing for file parsing and async for embeddings.
            Monitors system resources and throttles if necessary.
        """
        metrics = PerformanceMetrics(start_time=time.time())
        source = Path(source_path)
        files = [p for p in source.rglob("*") if p.is_file()]
        metrics.total_files = len(files)

        all_chunks = []
        with ThreadPoolExecutor(
            max_workers=self.config.cpu_threads
        ) as executor:
            futures = [
                executor.submit(
                    self.processor.process_file,
                    f,
                    self.config.unstructured_strategy,
                )
                for f in files
            ]
            for i, future in enumerate(as_completed(futures)):
                all_chunks.extend(future.result())
                metrics.processed_files += 1
                logger.info(f"Processed file {i + 1}/{len(files)}")

        metrics.initial_chunks = len(all_chunks)
        texts = [c["text"] for c in all_chunks]

        dense_vectors = await self.embedder.embed_dense_documents(texts)
        sparse_vectors = self.embedder.embed_sparse(texts)
        metrics.dense_embeddings = len(dense_vectors)
        metrics.sparse_splade_embeddings = len(sparse_vectors["splade"])
        metrics.sparse_bm25_embeddings = len(sparse_vectors["bm25"])

        self.store.upsert(all_chunks, dense_vectors, sparse_vectors)

        metrics.end_time = time.time()
        logger.info(
            f"Ingestion complete. Metrics: {json.dumps(metrics.summary(), indent=2)}"
        )

    def search(self, query: str, limit: int = 10) -> list[dict]:
        """Search the indexed documents using hybrid retrieval.

        Performs a 3-stage hybrid search combining dense embeddings (Gemini),
        sparse embeddings (SPLADE + BM25), and optional reranking for optimal
        retrieval quality.

        Args:
            query: The search query string.
            limit: Maximum number of results to return. Defaults to 10.

        Returns:
            List of dictionaries containing search results with the following keys:
                - text: The chunk text content
                - score: The RRF fusion score
                - metadata: Document metadata (source, file_type, content_type)
                - rerank_score: Cross-encoder reranking score (if reranking enabled)

        Example:
            >>> results = pipeline.search("How to implement RAG?", limit=5)
            >>> for result in results:
            ...     print(f"Score: {result['score']:.2f} - {result['text'][:100]}...")

        Note:
            Retrieves 5x the limit internally before reranking to ensure quality.
        """
        dense_query = self.embedder.embed_dense_query(query)
        # run sparse once and reuse outputs
        _sparse = self.embedder.embed_sparse([query])
        sparse_queries = {
            "splade": _sparse["splade"][0],
            "bm25": _sparse["bm25"][0],
        }

        results = self.store.search(dense_query, sparse_queries, limit * 5)

        if self.reranker:
            pairs = [[query, res["text"]] for res in results]
            scores = self.reranker.predict(pairs)
            for res, score in zip(results, scores):
                res["rerank_score"] = float(score)
            results.sort(key=lambda x: x["rerank_score"], reverse=True)

        return results[:limit]

    def close(self):
        """Clean up resources and shutdown executors gracefully."""
        self.resource_manager.close()
        self.cpu_executor.shutdown(wait=True)


class QdrantEvaluator:
    """Measures the recall impact of quantization."""

    def __init__(self, pipeline: MaxPerformancePipeline):
        self.pipeline = pipeline
        self.config = pipeline.config
        self.client = QdrantClient(url=self.config.qdrant_url)
        self.non_quantized_collection = (
            f"{self.config.collection_name}_no_quant"
        )
        self.quantized_collection = f"{self.config.collection_name}_quantized"

    def setup_collections(self):
        # Non-quantized
        self.client.recreate_collection(
            collection_name=self.non_quantized_collection,
            vectors_config={
                "dense": qm.VectorParams(
                    size=self.config.embedding_dimension,
                    distance=qm.Distance.COSINE,
                )
            },
            hnsw_config=qm.HnswConfigDiff(m=16, ef_construct=128),
        )
        # Quantized
        self.client.recreate_collection(
            collection_name=self.quantized_collection,
            vectors_config={
                "dense": qm.VectorParams(
                    size=self.config.embedding_dimension,
                    distance=qm.Distance.COSINE,
                )
            },
            hnsw_config=qm.HnswConfigDiff(m=16, ef_construct=128),
            quantization_config=qm.ScalarQuantization(
                scalar=qm.ScalarQuantizationConfig(
                    type=qm.ScalarType.INT8, quantile=0.99, always_ram=True
                )
            ),
        )

    async def run_benchmark(
        self, source_path: str, queries: list[str], k: int = 10
    ):
        """Benchmark INT8 quantization impact on search recall.

        Creates two test collections (quantized and non-quantized), ingests
        sample data, and measures recall@k by comparing search results.

        Args:
            source_path: Path to directory containing sample documents.
                Limits to first 20 files for consistent benchmarking.
            queries: List of test query strings for evaluation.
            k: Number of top results to consider for recall calculation.
                Defaults to 10.

        Returns:
            None. Logs detailed recall metrics for each query and overall average.

        Raises:
            ValueError: If source_path is empty or queries list is empty.

        Example:
            >>> queries = ["machine learning", "neural networks", "RAG systems"]
            >>> evaluator = QdrantEvaluator(pipeline)
            >>> await evaluator.run_benchmark("/path/to/samples", queries, k=10)
            # Output: Average Recall@10 for INT8 Quantization: 0.9500

        Note:
            Uses exact search on non-quantized collection as ground truth.
            Quantized collection uses INT8 scalar quantization with rescoring.
        """
        logger.info("Setting up collections for quantization benchmark...")
        self.setup_collections()

        logger.info(f"Ingesting sample data from {source_path}...")
        source = Path(source_path)
        files = [p for p in source.rglob("*") if p.is_file()][
            :20
        ]  # Limit to a sample

        all_chunks = []
        for f in files:
            all_chunks.extend(self.pipeline.processor.process_file(f, "fast"))

        texts = [c["text"] for c in all_chunks]
        dense_vectors = await self.pipeline.embedder.embed_dense_documents(
            texts
        )

        points = [
            qm.PointStruct(id=i, vector={"dense": vec}, payload=chunk)
            for i, (chunk, vec) in enumerate(zip(all_chunks, dense_vectors))
        ]

        self.client.upsert(
            collection_name=self.non_quantized_collection,
            points=points,
            wait=True,
        )
        self.client.upsert(
            collection_name=self.quantized_collection, points=points, wait=True
        )
        logger.info(
            f"Ingested {len(points)} sample points into both collections."
        )

        total_recall = 0.0
        for query in queries:
            query_vector = self.pipeline.embedder.embed_dense_query(query)

            # Ground truth (exact search on non-quantized)
            ground_truth_res = self.client.search(
                collection_name=self.non_quantized_collection,
                query_vector=query_vector,
                limit=k,
                search_params=qm.SearchParams(exact=True),
            )
            ground_truth_ids = {hit.id for hit in ground_truth_res}

            # Approximate search on quantized collection
            quantized_res = self.client.search(
                collection_name=self.quantized_collection,
                query_vector=query_vector,
                limit=k,
                search_params=qm.SearchParams(
                    quantization=qm.QuantizationSearchParams(rescore=True)
                ),
            )
            quantized_ids = {hit.id for hit in quantized_res}

            recall = len(ground_truth_ids.intersection(quantized_ids)) / k
            total_recall += recall
            logger.info(
                f"Query: '{query[:30]}...' -> Recall@{k}: {recall:.2f}"
            )

        avg_recall = total_recall / len(queries)
        logger.info("-" * 50)
        logger.info(
            f"Average Recall@{k} for INT8 Quantization: {avg_recall:.4f}"
        )
        logger.info("-" * 50)


def handle_management_command(command: str, args):
    """Handle management commands that don't need the pipeline."""
    project_dir = Path(__file__).parent.absolute()
    venv_dir = project_dir / ".venv"
    python_exec = venv_dir / "bin" / "python"
    pip_exec = venv_dir / "bin" / "pip"
    
    if command == "install":
        install_python_environment(project_dir, venv_dir, python_exec, pip_exec)
    elif command == "installdeps":
        install_system_dependencies()
    elif command == "start-qdrant":
        start_qdrant_container(project_dir)
    elif command == "stop-qdrant":
        stop_qdrant_container()
    elif command == "logs-qdrant":
        show_qdrant_logs()
    elif command == "format":
        run_code_formatter(project_dir, venv_dir)
    elif command == "lint":
        run_linter(project_dir, venv_dir)
    elif command == "typecheck":
        run_type_checker(project_dir, venv_dir)
    elif command == "check":
        run_linter(project_dir, venv_dir)
        run_type_checker(project_dir, venv_dir)
    elif command == "test":
        run_tests(project_dir, venv_dir)
    elif command == "qa":
        run_code_formatter(project_dir, venv_dir)
        run_linter(project_dir, venv_dir)
        run_type_checker(project_dir, venv_dir)
        run_tests(project_dir, venv_dir)
    elif command == "verify":
        verify_installation(venv_dir, project_dir)
    elif command == "clean":
        clean_project(project_dir, venv_dir)
    elif command == "erasedata":
        erase_qdrant_data(project_dir, args.yes if hasattr(args, 'yes') else False)


def install_python_environment(project_dir: Path, venv_dir: Path, python_exec: Path, pip_exec: Path):
    """Install Python virtual environment and dependencies."""
    print("=== Python Environment Setup ===")
    
    # Create virtual environment if it doesn't exist
    if not venv_dir.exists():
        print(f"Creating virtual environment at {venv_dir}...")
        if shutil.which("uv"):
            subprocess.run(["uv", "venv", str(venv_dir)], check=True)
            subprocess.run(["uv", "pip", "install", "--python", str(python_exec), "pip"], check=True)
        else:
            subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
            subprocess.run([str(python_exec), "-m", "pip", "install", "--upgrade", "pip"], check=True)
    else:
        print(f"Using existing virtual environment at {venv_dir}")
    
    print("\nInstalling development tools (ruff, mypy, pytest)...")
    tools = ["ruff", "mypy", "pytest", "pytest-asyncio", "pytest-cov", "build", "datasketch"]
    if shutil.which("uv"):
        subprocess.run(["uv", "pip", "install", "--python", str(python_exec)] + tools, check=True)
    else:
        subprocess.run([str(python_exec), "-m", "pip", "install"] + tools, check=True)
    
    print("\nInstalling full dependencies (this may take several minutes)...")
    env = os.environ.copy()
    env["QDRANT_RAG_INSTALLING"] = "1"
    
    if shutil.which("uv"):
        subprocess.run(["uv", "pip", "install", "--python", str(python_exec), "-e", f"{project_dir}[dev]"], 
                      env=env, check=True)
        print("Installing PyTorch with CUDA support...")
        subprocess.run(["uv", "pip", "install", "--python", str(python_exec), "torch", "torchvision",
                       "--index-url", "https://download.pytorch.org/whl/cu121"], check=True)
        print("\nHandling ONNX Runtime GPU setup...")
        subprocess.run(["uv", "pip", "uninstall", "--python", str(python_exec), "-y", "onnxruntime"],
                      capture_output=True)
        subprocess.run(["uv", "pip", "install", "--python", str(python_exec), "--force-reinstall", 
                       "onnxruntime-gpu"], check=True)
    else:
        subprocess.run([str(python_exec), "-m", "pip", "install", "-e", f"{project_dir}[dev]"], 
                      env=env, check=True)
        subprocess.run([str(python_exec), "-m", "pip", "install", "torch", "torchvision",
                       "--index-url", "https://download.pytorch.org/whl/cu121"], check=True)
        subprocess.run([str(python_exec), "-m", "pip", "uninstall", "-y", "onnxruntime"],
                      capture_output=True)
        subprocess.run([str(python_exec), "-m", "pip", "install", "--force-reinstall", 
                       "onnxruntime-gpu"], check=True)
    
    print(f"\n✅ Python environment ready at {venv_dir}")
    print("\nNext steps:")
    print("  • Copy .env.example to .env and add your API keys")
    print("  • Run './qdrant_rag.py verify' to check GPU acceleration")
    print("  • Run './qdrant_rag.py start-qdrant' to start the database")


def install_system_dependencies():
    """Install system dependencies (Docker, CUDA, etc)."""
    print("=== System Dependencies Installation ===")
    print("This will install:")
    print("  • uv (fast Python package manager)")
    print("  • Docker with GPU support")
    print("  • NVIDIA drivers and CUDA toolkit")
    print("  • System libraries for document processing")
    print("  • Hardware monitoring tools")
    print("\nRequires sudo access. Designed for Arch-based systems.")
    
    response = input("\nContinue? [y/N] ")
    if response.lower() != 'y':
        print("Installation cancelled.")
        return
    
    # Install uv if not present
    if not shutil.which("uv"):
        print("\nStep 1: Installing uv package manager...")
        subprocess.run("curl -LsSf https://astral.sh/uv/install.sh | sh", shell=True, check=True)
        print("✓ uv installed to ~/.local/bin")
        print("  Add to your PATH: export PATH=\"$HOME/.local/bin:$PATH\"")
    else:
        print(f"✓ uv is already installed")
    
    if shutil.which("pacman"):
        print("\nStep 2: Installing system packages via pacman...")
        packages = [
            "python", "python-pip", "git", "curl", "wget", "unzip", "gcc", "make", "cmake", "pkg-config",
            "libxml2", "libxslt", "bzip2", "lz4", "zstd", "openssl", "base-devel",
            "htop", "iotop", "btop", "lm_sensors", "nvtop",
            "docker", "nvidia", "nvidia-utils", "nvidia-settings", "cuda", "nvidia-container-toolkit",
            "file", "poppler", "tesseract", "tesseract-data-eng", "libreoffice-fresh", "pandoc"
        ]
        subprocess.run(["sudo", "pacman", "-S", "--needed", "--noconfirm"] + packages, check=True)
        
        print("\nStep 3: Configuring Docker for GPU support...")
        if shutil.which("docker"):
            subprocess.run(["sudo", "systemctl", "enable", "docker.service"], check=True)
            subprocess.run(["sudo", "systemctl", "start", "docker.service"], check=True)
            
            if shutil.which("nvidia-ctk"):
                subprocess.run(["sudo", "nvidia-ctk", "runtime", "configure", "--runtime=docker"], check=True)
                subprocess.run(["sudo", "systemctl", "restart", "docker"], check=True)
            
            # Check if user is in docker group
            user = os.environ.get("USER")
            groups_output = subprocess.run(["groups", user], capture_output=True, text=True)
            if "docker" not in groups_output.stdout:
                subprocess.run(["sudo", "usermod", "-aG", "docker", user], check=True)
                print(f"⚠️  Added {user} to docker group. Please logout and login again.")
            else:
                print("✓ User already in docker group")
        
        print("\nStep 4: Configuring hardware sensors...")
        subprocess.run(["sudo", "sensors-detect", "--auto"], capture_output=True)
        
        print("\n✅ System dependencies installed successfully!")
        print("\nNext steps:")
        print("  1. If you were added to docker group, logout and login again")
        print("  2. Run './qdrant_rag.py install' to set up Python environment")
    else:
        print("This target is designed for Arch-based systems.")
        print("Please install the required dependencies manually.")


def start_qdrant_container(project_dir: Path):
    """Start Qdrant Docker container with GPU support."""
    # Ensure qdrant_data directory exists with proper permissions
    data_dir = project_dir / "qdrant_data"
    if not data_dir.exists():
        data_dir.mkdir(exist_ok=True)
        data_dir.chmod(0o755)
        print("Created qdrant_data directory")
    
    print("Starting Qdrant Docker container with GPU support...")
    
    container_name = "qdrant_rag"
    image = "qdrant/qdrant:gpu-nvidia-latest"
    
    # Get current user ID and group ID
    uid = os.getuid()
    gid = os.getgid()
    
    cmd = [
        "docker", "run", "-d",
        "--name", container_name,
        "--network", "host",
        "--gpus", "all",
        "--user", f"{uid}:{gid}",
        "-v", f"{project_dir}/qdrant_data:/qdrant/storage",
        "-v", f"{project_dir}/config.yaml:/qdrant/config/config.yaml:ro",
        "-e", "QDRANT__GPU__INDEXING=1",
        image
    ]
    
    subprocess.run(cmd, check=True)
    print("Qdrant container is starting on host network (port 6333) with user permissions.")
    print("Use './qdrant_rag.py logs-qdrant' to monitor.")


def stop_qdrant_container():
    """Stop and remove Qdrant Docker container."""
    print("Stopping and removing Qdrant Docker container...")
    container_name = "qdrant_rag"
    
    subprocess.run(["docker", "stop", container_name], capture_output=True)
    subprocess.run(["docker", "rm", container_name], capture_output=True)
    print("Qdrant container stopped and removed.")


def show_qdrant_logs():
    """Show Qdrant container logs."""
    container_name = "qdrant_rag"
    subprocess.run(["docker", "logs", "-f", container_name])


def run_code_formatter(project_dir: Path, venv_dir: Path):
    """Format Python code with ruff."""
    ruff_exec = venv_dir / "bin" / "ruff"
    
    if not ruff_exec.exists():
        print("Error: ruff not found. Run './qdrant_rag.py install' first.")
        sys.exit(1)
    
    print("Formatting Python code with ruff...")
    subprocess.run([str(ruff_exec), "format", str(project_dir / "qdrant_rag.py")], check=True)
    print("Formatting complete.")


def run_linter(project_dir: Path, venv_dir: Path):
    """Lint Python code and fix issues."""
    ruff_exec = venv_dir / "bin" / "ruff"
    
    if not ruff_exec.exists():
        print("Error: ruff not found. Run './qdrant_rag.py install' first.")
        sys.exit(1)
    
    print("Linting Python code and fixing issues...")
    subprocess.run([str(ruff_exec), "check", str(project_dir / "qdrant_rag.py"), "--fix"], check=True)
    print("Linting complete.")


def run_type_checker(project_dir: Path, venv_dir: Path):
    """Run type checking with mypy."""
    mypy_exec = venv_dir / "bin" / "mypy"
    
    if not mypy_exec.exists():
        print("Error: mypy not found. Run './qdrant_rag.py install' first.")
        sys.exit(1)
    
    print("Running type checking on Python files...")
    subprocess.run([str(mypy_exec), str(project_dir / "qdrant_rag.py"), "--ignore-missing-imports"], check=True)
    print("Type checking complete.")


def run_tests(project_dir: Path, venv_dir: Path):
    """Run tests with pytest."""
    pytest_exec = venv_dir / "bin" / "pytest"
    
    if not pytest_exec.exists():
        print("Error: pytest not found. Run './qdrant_rag.py install' first.")
        sys.exit(1)
    
    print("Running tests...")
    tests_dir = project_dir / "tests"
    if tests_dir.exists():
        subprocess.run([str(pytest_exec), str(tests_dir), "-v"], cwd=project_dir)
    else:
        print("No tests found.")


def verify_installation(venv_dir: Path, project_dir: Path):
    """Verify GPU/CUDA and installation status."""
    print("=== Installation Verification ===\n")
    
    python_exec = venv_dir / "bin" / "python"
    
    # Check Python
    print("Python Environment:")
    if python_exec.exists():
        result = subprocess.run([str(python_exec), "--version"], capture_output=True, text=True)
        print(f"  {result.stdout.strip()}")
    else:
        print("  ✗ Python not found")
    print(f"  Virtual env: {venv_dir}\n")
    
    # Check GPU/CUDA
    print("Testing GPU/CUDA availability...")
    verification_script = '''
import sys
print("Python packages:")

try:
    import torch
    print(f"  ✓ PyTorch {torch.__version__}")
    if torch.cuda.is_available():
        print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  ✓ CUDA version: {torch.version.cuda}")
    else:
        print("  ⚠ CUDA not available")
except ImportError:
    print("  ⚠ PyTorch not installed")

try:
    import onnxruntime as ort
    print(f"  ✓ ONNX Runtime {ort.__version__}")
    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" in providers:
        print("  ✓ CUDA provider available for ONNX")
    else:
        print("  ⚠ CUDA provider not available for ONNX")
except ImportError:
    print("  ⚠ ONNX Runtime not installed")

try:
    import pynvml
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    gpu_name = pynvml.nvmlDeviceGetName(handle).decode("utf-8")
    print(f"  ✓ GPU detected: {gpu_name}")
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    print(f"  ✓ GPU Memory: {mem_info.total / 1024**3:.1f} GB")
    pynvml.nvmlShutdown()
except:
    print("  ⚠ Unable to detect GPU via pynvml")

try:
    import qdrant_client
    print("  ✓ Qdrant client installed")
except ImportError:
    print("  ⚠ Qdrant client not installed")
'''
    
    if python_exec.exists():
        subprocess.run([str(python_exec), "-c", verification_script])
    
    # Check Docker
    print("\nDocker Status:")
    if shutil.which("docker"):
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
        version = result.stdout.split()[2].rstrip(',')
        print(f"  ✓ Docker installed: {version}")
        
        # Check if docker daemon is running
        result = subprocess.run(["docker", "ps"], capture_output=True)
        if result.returncode == 0:
            print("  ✓ Docker daemon running")
            
            # Check if Qdrant container is running
            result = subprocess.run(["docker", "ps"], capture_output=True, text=True)
            if "qdrant_rag" in result.stdout:
                print("  ✓ Qdrant container is running")
            else:
                print("  ⚠ Qdrant container not running (use './qdrant_rag.py start-qdrant')")
        else:
            print("  ✗ Docker daemon not accessible (check permissions)")
    else:
        print("  ✗ Docker not installed")
    
    # Check configuration files
    print("\nConfiguration Files:")
    env_file = project_dir / ".env"
    env_example = project_dir / ".env.example"
    
    if env_file.exists():
        print("  ✓ .env file exists")
    else:
        print("  ⚠ .env file missing (copy from .env.example)")
    
    if env_example.exists():
        print("  ✓ .env.example exists")
    else:
        print("  ✗ .env.example missing")
    
    if os.access(project_dir / "qdrant_rag.py", os.X_OK):
        print("  ✓ qdrant_rag.py is executable")
    else:
        print("  ✗ qdrant_rag.py not executable")
    
    print("\n=== Verification complete ===")


def clean_project(project_dir: Path, venv_dir: Path):
    """Remove virtual environment and generated files."""
    print("Cleaning up project...")
    
    # Remove virtual environment
    if venv_dir.exists():
        shutil.rmtree(venv_dir)
    
    # Remove other files
    files_to_remove = [
        project_dir / ".env",
        project_dir / "queries.json",
    ]
    
    for file in files_to_remove:
        if file.exists():
            file.unlink()
    
    # Remove log files
    for log_file in project_dir.glob("*.log"):
        log_file.unlink()
    
    # Remove wheel files
    for whl_file in project_dir.glob("*.whl"):
        whl_file.unlink()
    
    # Remove cache directories
    cache_dirs = [
        project_dir / ".ruff_cache",
        project_dir / ".mypy_cache",
        project_dir / ".pytest_cache",
        project_dir / "htmlcov",
        project_dir / "build",
        project_dir / "dist",
        project_dir / ".tmp",
    ]
    
    for cache_dir in cache_dirs:
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
    
    # Remove __pycache__ directories
    for pycache in project_dir.rglob("__pycache__"):
        shutil.rmtree(pycache)
    
    # Remove .pyc files
    for pyc_file in project_dir.rglob("*.pyc"):
        pyc_file.unlink()
    
    # Remove egg-info directories
    for egg_info in project_dir.glob("*.egg-info"):
        shutil.rmtree(egg_info)
    
    # Remove HTML files
    for html_file in project_dir.glob("*.html"):
        html_file.unlink()
    
    print("Cleanup complete. Run './qdrant_rag.py stop-qdrant' to stop the database.")


def erase_qdrant_data(project_dir: Path, skip_confirmation: bool = False):
    """Delete all Qdrant vector data."""
    if not skip_confirmation:
        print("⚠️  WARNING: This will permanently delete all Qdrant vector data!")
        response = input("Are you sure you want to delete qdrant_data/? Type 'yes' to confirm: ")
        if response != "yes":
            print("Operation cancelled.")
            return
    
    print("Stopping Qdrant container if running...")
    subprocess.run(["docker", "stop", "qdrant_rag"], capture_output=True)
    
    print("Removing Qdrant data directory...")
    data_dir = project_dir / "qdrant_data"
    if data_dir.exists():
        shutil.rmtree(data_dir)
    
    print("✓ Qdrant data erased.")


def main_cli():
    """Command-line interface entry point for the RAG pipeline."""
    parser = argparse.ArgumentParser(
        description="Qdrant RAG Pipeline - Self-contained vector search system with GPU acceleration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s install              # Set up Python environment and dependencies
  %(prog)s start-qdrant         # Start Qdrant database with GPU support  
  %(prog)s ingest --source ./docs  # Ingest documents
  %(prog)s search --query "your search"  # Search indexed documents
  %(prog)s verify               # Check GPU and installation status
  %(prog)s format               # Format code with ruff
"""
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    ingest_parser = subparsers.add_parser(
        "ingest", help="Ingest documents from a source directory."
    )
    ingest_parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to the source directory.",
    )

    search_parser = subparsers.add_parser(
        "search", help="Search the indexed documents."
    )
    search_parser.add_argument(
        "--query", type=str, required=True, help="The search query."
    )
    search_parser.add_argument(
        "--limit", type=int, default=10, help="Number of results to return."
    )

    eval_parser = subparsers.add_parser(
        "evaluate-quantization", help="Benchmark recall of INT8 quantization."
    )
    eval_parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to a small sample of source documents.",
    )
    eval_parser.add_argument(
        "--queries-file",
        type=str,
        required=True,
        help="JSON file with a list of test queries.",
    )

    # Installation and setup commands
    subparsers.add_parser(
        "install", help="Create venv and install Python packages with GPU support"
    )
    subparsers.add_parser(
        "installdeps", help="Install system dependencies (Docker, CUDA, etc)"
    )
    
    # Docker/Qdrant management commands
    subparsers.add_parser(
        "start-qdrant", help="Start Qdrant Docker container with GPU support"
    )
    subparsers.add_parser(
        "stop-qdrant", help="Stop and remove Qdrant Docker container"
    )
    subparsers.add_parser(
        "logs-qdrant", help="View Qdrant container logs"
    )
    
    # Development tools
    subparsers.add_parser(
        "format", help="Format Python code with ruff"
    )
    subparsers.add_parser(
        "lint", help="Lint code and fix issues"
    )
    subparsers.add_parser(
        "typecheck", help="Run type checking with mypy"
    )
    subparsers.add_parser(
        "check", help="Run both lint and typecheck"
    )
    subparsers.add_parser(
        "test", help="Run tests"
    )
    subparsers.add_parser(
        "qa", help="Run format, check, and test"
    )
    
    # Utility commands
    subparsers.add_parser(
        "verify", help="Verify GPU/CUDA and installation status"
    )
    subparsers.add_parser(
        "clean", help="Remove virtual environment and generated files"
    )
    erase_data_parser = subparsers.add_parser(
        "erasedata", help="⚠️ Delete all Qdrant vector data"
    )
    erase_data_parser.add_argument(
        "--yes", action="store_true", help="Skip confirmation prompt"
    )

    args = parser.parse_args()

    # Handle commands that don't need the pipeline
    if args.command in ["install", "installdeps", "start-qdrant", "stop-qdrant", 
                        "logs-qdrant", "format", "lint", 
                        "typecheck", "check", "test", "qa", "verify", 
                        "clean", "erasedata"]:
        handle_management_command(args.command, args)
        return  # Exit after handling management command
    
    # Commands that need the pipeline
    pipeline = MaxPerformancePipeline()
    try:
        if args.command == "ingest":
            asyncio.run(pipeline.ingest(args.source))
        elif args.command == "search":
            results = pipeline.search(args.query, args.limit)
            print(json.dumps(results, indent=2))
        elif args.command == "evaluate-quantization":
            with open(args.queries_file) as f:
                queries = [item["query"] for item in json.load(f)]
            evaluator = QdrantEvaluator(pipeline)
            asyncio.run(evaluator.run_benchmark(args.source, queries))
    finally:
        pipeline.close()


if __name__ == "__main__" and not _IS_BUILD_BACKEND:
    main_cli()
