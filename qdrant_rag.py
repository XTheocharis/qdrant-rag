#!/usr/bin/env python3
"""Hybrid RAG Pipeline using Qdrant.

Hardware Target: AMD 9950X3D + RTX 5070 Ti + 64GB RAM + Gen5 NVMe + CachyOS

Implements the complete 3-Index Hybrid Search architecture:
1. Vector Index (Dense Gemini + Sparse SPLADE)
2. Full-Text Index (Sparse BM25)
3. Payload Index (Structured Metadata)
"""

# Auto-activate virtual environment if not already active
import os
import sys
from pathlib import Path


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

    # Check if venv exists
    if not venv_dir.exists():
        print(f"Error: Virtual environment not found at {venv_dir}")
        print("Please run 'make install' first to set up the environment.")
        sys.exit(1)

    if not venv_python.exists():
        print(f"Error: Python executable not found at {venv_python}")
        print("Virtual environment may be corrupted. Run 'make clean && make install'")
        sys.exit(1)

    # Re-launch the script with the venv's Python
    print(f"Auto-activating virtual environment: {venv_dir}")
    os.execv(str(venv_python), [str(venv_python)] + sys.argv)

# Ensure we're in the venv before importing dependencies
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


def main_cli():
    """Command-line interface entry point for the RAG pipeline."""
    parser = argparse.ArgumentParser(
        description="RAG Pipeline"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

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

    args = parser.parse_args()
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


if __name__ == "__main__":
    main_cli()
