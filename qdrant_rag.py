#!/usr/bin/env python3
"""
High-Performance Hybrid RAG Pipeline using Qdrant

Hardware Target: AMD 9950X3D + RTX 5070 Ti + 64GB RAM + Gen5 NVMe + CachyOS

Implements the complete 3-Index Hybrid Search architecture:
1. Vector Index (Dense Gemini + Sparse SPLADE)
2. Full-Text Index (Sparse BM25)
3. Payload Index (Structured Metadata)
"""

import os
import sys
import logging
import time
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone
from enum import Enum
import json
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import threading
import gc
import psutil
import signal
import re
import unicodedata
import uuid
import argparse

# --- Core Dependencies ---
from google import genai
from google.genai.types import EmbedContentConfig
import numpy as np
import torch
import pynvml
import onnxruntime as ort
from sklearn.metrics import recall_score

# --- Qdrant ---
from qdrant_client import QdrantClient, models as qm

# --- Document Parsing & Chunking ---
from unstructured.partition.auto import partition as unstructured_partition
from llama_index.core.node_parser import SemanticSplitterNodeParser, CodeSplitter
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding as LlamaGeminiEmbedding
from llama_index.core import Document as LlamaDocument
from text_dedup.minhash import MinHashLSH

# --- Embeddings & Reranking ---
from fastembed import SparseTextEmbedding
from sentence_transformers import CrossEncoder

# --- Optional Dependency Checks ---
try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

# --- Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('qdrant_rag.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)


# --- Enums and Dataclasses ---
@dataclass
class PerformanceMetrics:
    """Track performance across the pipeline"""
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

    def summary(self) -> Dict[str, Any]:
        total_time = self.total_time
        return {
            "total_time_seconds": round(total_time, 2),
            "files_per_second": round(self.processed_files / total_time, 2) if total_time > 0 else 0,
            "chunks_per_second": round(self.deduped_chunks / total_time, 2) if total_time > 0 else 0,
            **self.__dict__
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
        self.max_cpu_temp_c = float(os.getenv('MAX_CPU_TEMP', 85))
        self.max_gpu_temp_c = float(os.getenv('MAX_GPU_TEMP', 83))
        self.critical_cpu_temp_c = float(os.getenv('CRITICAL_CPU_TEMP', 95))
        self.critical_gpu_temp_c = float(os.getenv('CRITICAL_GPU_TEMP', 90))

        self.throttle_delay = 0.1
        self.recovery_threshold = 0.8

        self.temp_monitoring_available = self._check_temp_monitoring()
        self._set_process_niceness()
        self.setup_signal_handlers()

    def _check_temp_monitoring(self) -> Dict[str, bool]:
        available = {'cpu': False, 'gpu': False}
        try:
            if hasattr(psutil, 'sensors_temperatures') and psutil.sensors_temperatures():
                available['cpu'] = True
        except Exception: pass
        try:
            pynvml.nvmlInit()
            available['gpu'] = True
            pynvml.nvmlShutdown()
        except Exception: pass
        logger.info(f"Temperature monitoring available: {available}")
        return available

    def get_cpu_temperature(self) -> Optional[float]:
        if not self.temp_monitoring_available['cpu']: return None
        try:
            temps = psutil.sensors_temperatures()
            if 'k10temp' in temps: # AMD
                tctl = next((s for s in temps['k10temp'] if 'Tctl' in s.label), None)
                return tctl.current if tctl else max(s.current for s in temps['k10temp'])
            elif 'coretemp' in temps: # Intel
                return max(s.current for s in temps['coretemp'])
        except Exception as e: logger.debug(f"CPU temp reading failed: {e}")
        return None

    def get_gpu_temperature(self) -> Optional[float]:
        if not self.temp_monitoring_available['gpu']: return None
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            pynvml.nvmlShutdown()
            return float(temp)
        except Exception as e: logger.debug(f"GPU temp reading failed: {e}")
        return None

    def _set_process_niceness(self):
        try:
            os.nice(5)
            logger.info(f"Process niceness set to {os.nice(0)} for balanced performance.")
        except Exception as e: logger.warning(f"Could not set process niceness: {e}")

    def get_system_stats(self) -> Dict[str, Any]:
        try:
            memory = psutil.virtual_memory()
            return {
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "mem_percent": memory.percent,
                "mem_available_gb": memory.available / (1024**3),
                "cpu_temp_c": self.get_cpu_temperature(),
                "gpu_temp_c": self.get_gpu_temperature(),
            }
        except Exception: return {}

    def should_throttle(self) -> Tuple[bool, str]:
        stats = self.get_system_stats()
        if not stats: return False, ""
        
        if (temp := stats.get("cpu_temp_c")) and temp > self.critical_cpu_temp_c: raise RuntimeError(f"CPU temp critical: {temp}°C")
        if (temp := stats.get("gpu_temp_c")) and temp > self.critical_gpu_temp_c: raise RuntimeError(f"GPU temp critical: {temp}°C")

        if (temp := stats.get("cpu_temp_c")) and temp > self.max_cpu_temp_c: return True, f"CPU Temp High ({temp}°C)"
        if (temp := stats.get("gpu_temp_c")) and temp > self.max_gpu_temp_c: return True, f"GPU Temp High ({temp}°C)"
        if stats["cpu_percent"] > self.max_cpu_percent: return True, f"CPU High ({stats['cpu_percent']}%)"
        if stats["mem_percent"] > self.max_memory_percent: return True, f"Memory High ({stats['mem_percent']}%)"
        if stats["mem_available_gb"] < self.min_free_memory_gb: return True, f"Memory Low ({stats['mem_available_gb']:.1f}GB)"
        
        return False, ""

    async def adaptive_throttle(self, operation_name: str):
        if not self.monitoring: return
        should, reason = self.should_throttle()
        with self.throttle_lock:
            if should and not self.emergency_throttle:
                self.emergency_throttle = True
                logger.warning(f"THROTTLING {operation_name}: {reason}")
            elif not should and self.emergency_throttle:
                stats = self.get_system_stats()
                if (stats.get("cpu_percent", 100) < self.max_cpu_percent * self.recovery_threshold and
                    stats.get("mem_percent", 100) < self.max_memory_percent * self.recovery_threshold):
                    self.emergency_throttle = False
                    logger.info(f"RECOVERY: {operation_name} throttling disabled.")
        if self.emergency_throttle:
            await asyncio.sleep(self.throttle_delay)

    def setup_signal_handlers(self):
        def handler(signum, frame):
            logger.info(f"Signal {signum} received, shutting down.")
            self.monitoring = False
        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)

    def close(self):
        self.monitoring = False
        logger.info("Resource manager shut down.")


# --- Core Pipeline Components ---
class MaxPerformanceConfig:
    """Configuration optimized for the target hardware."""
    def __init__(self):
        self.gemini_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not self.gemini_api_key: raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY not set")
        
        self.cpu_threads = 32
        self.gemini_concurrent_requests = 32
        self.gemini_batch_size = 100
        
        self.embedding_model = os.getenv("EMBEDDING_MODEL", "gemini-embedding-001")
        self.embedding_dimension = 3072
        
        self.splade_model = "prithivida/Splade_PP_en_v1"
        self.bm25_model = "Qdrant/bm25"
        
        self.qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        self.collection_name = "max_performance_hybrid"
        
        self.enable_gpu = os.getenv("ENABLE_GPU_ACCELERATION", "true").lower() == "true"
        self.unstructured_strategy = os.getenv("UNSTRUCTURED_STRATEGY", "hi_res")
        self.reranker_model = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
        self.enable_reranking = True

class Embedder:
    """Handles dense (Gemini) and double sparse (SPLADE + BM25) embeddings."""
    def __init__(self, config: MaxPerformanceConfig, executor: ThreadPoolExecutor):
        self.config = config
        self.semaphore = asyncio.Semaphore(config.gemini_concurrent_requests)
        self.cpu_executor = executor
        
        genai.configure(api_key=config.gemini_api_key)
        self.gemini_client = genai.Client(api_key=config.gemini_api_key)
        self.llama_gemini_embedder = LlamaGeminiEmbedding(
            api_key=config.gemini_api_key,
            model_name=config.embedding_model
        )
        
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if config.enable_gpu else ["CPUExecutionProvider"]
        self.splade_embedder = SparseTextEmbedding(model_name=config.splade_model, providers=providers)
        self.bm25_embedder = SparseTextEmbedding(model_name=config.bm25_model, providers=providers)
        self._verify_fastembed_gpu()

    def _verify_fastembed_gpu(self):
        if self.config.enable_gpu:
            if 'CUDAExecutionProvider' not in ort.get_available_providers():
                logger.warning("ONNX Runtime CUDA provider not found. FastEmbed will use CPU.")
                return
            
            splade_providers = self.splade_embedder.model.model.get_providers()
            bm25_providers = self.bm25_embedder.model.model.get_providers()

            if 'CUDAExecutionProvider' in splade_providers:
                logger.info("SUCCESS: FastEmbed SPLADE is actively using GPU (CUDAExecutionProvider).")
            else:
                logger.warning(f"FastEmbed SPLADE is NOT using GPU. Active providers: {splade_providers}")

            if 'CUDAExecutionProvider' in bm25_providers:
                logger.info("SUCCESS: FastEmbed BM25 is actively using GPU (CUDAExecutionProvider).")
            else:
                logger.warning(f"FastEmbed BM25 is NOT using GPU. Active providers: {bm25_providers}")

    async def _rate_limited_gemini_batch_request(self, texts: List[str], task_type: str) -> List[List[float]]:
        async with self.semaphore:
            loop = asyncio.get_event_loop()
            def embed_batch():
                # Use embed_content with a list for batching per SDK docs
                res = self.gemini_client.models.embed_content(
                    model=self.config.embedding_model,
                    contents=texts,
                    config=EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=self.config.embedding_dimension
                    ),
                )
                return [emb.values for emb in res.embeddings]
            
            for attempt in range(5):
                try:
                    return await loop.run_in_executor(self.cpu_executor, embed_batch)
                except Exception as e:
                    if "rate limit" in str(e).lower() and attempt < 4:
                        await asyncio.sleep(2 ** attempt)
                    else:
                        raise

    async def embed_dense_documents(self, texts: List[str]) -> List[List[float]]:
        batch_size = self.config.gemini_batch_size
        tasks = [self._rate_limited_gemini_batch_request(texts[i:i+batch_size], "RETRIEVAL_DOCUMENT") for i in range(0, len(texts), batch_size)]
        results = await asyncio.gather(*tasks)
        return [item for sublist in results for item in sublist]

    def embed_dense_query(self, text: str) -> List[float]:
        res = self.gemini_client.models.embed_content(
            model=self.config.embedding_model,
            contents=[text],
            config=EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=self.config.embedding_dimension
            ),
        )
        return res.embeddings[0].values

    def embed_sparse(self, texts: List[str]) -> Dict[str, List[Dict[str, Any]]]:
        splade_embs = self.splade_embedder.embed(texts)
        bm25_embs = self.bm25_embedder.embed(texts)
        return {
            "splade": [{"indices": emb.indices.tolist(), "values": emb.values.tolist()} for emb in splade_embs],
            "bm25": [{"indices": emb.indices.tolist(), "values": emb.values.tolist()} for emb in bm25_embs]
        }

class StructuredDocumentSplitter:
    """XML/HTML-aware document splitter that respects document structure"""
    def split_html_document(self, html: str) -> list[str]:
        if not BS4_AVAILABLE: return [html]
        soup = BeautifulSoup(html, "html.parser")
        blocks = [tag.get_text(" ", strip=True) for tag in soup.find_all(["h1","h2","h3","p","li","pre","code"]) if tag.get_text(strip=True)]
        return blocks

    def split_xml_document(self, xml: str) -> list[str]:
        if not BS4_AVAILABLE: return [xml]
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
            breakpoint_percentile_threshold=95
        )
        self.code_splitter = CodeSplitter.from_defaults(language="python")

    def _normalize_and_clean(self, text: str) -> str:
        return unicodedata.normalize('NFC', text)

    def _deduplicate_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not chunks: return []
        lsh = MinHashLSH(threshold=0.85, num_perm=128)
        texts_to_check = [c['text'] for c in chunks]
        if not texts_to_check: return []
        
        is_duplicate_flags = lsh.find_duplicates(texts_to_check)
        
        unique_chunks = [chunks[i] for i, is_duplicate in enumerate(is_duplicate_flags) if not is_duplicate]
        logger.info(f"Deduplication removed {len(chunks) - len(unique_chunks)} near-duplicate chunks.")
        return unique_chunks

    def process_file(self, file_path: Path, unstructured_strategy: str) -> List[Dict[str, Any]]:
        ext = file_path.suffix.lower()
        
        chunks_text = []
        metadata = {"source": str(file_path), "file_type": ext}

        if ext in {'.py', '.js', '.java', '.cpp', '.c', '.h', '.rs', '.go'}: # Code
            content = self._normalize_and_clean(file_path.read_text(encoding='utf-8', errors='ignore'))
            lang_map = {'.py': 'python', '.js': 'javascript', '.java': 'java', '.cpp': 'cpp', '.c': 'c', '.h': 'c', '.rs': 'rust', '.go': 'go'}
            self.code_splitter.language = lang_map.get(ext, 'python')
            chunks_text = self.code_splitter.split_text(content)
            metadata["content_type"] = "code"
        elif ext in {'.html', '.htm', '.xhtml', '.xml', '.svg'}: # Markup
            content = self._normalize_and_clean(file_path.read_text(encoding='utf-8', errors='ignore'))
            chunks_text = self.structured_splitter.split_html_document(content) if ext.startswith('.htm') else self.structured_splitter.split_xml_document(content)
            metadata["content_type"] = "markup"
        else: # Documents
            elements = unstructured_partition(file_path=str(file_path), strategy=unstructured_strategy)
            content = self._normalize_and_clean("\n\n".join([el.text for el in elements if hasattr(el, 'text')]))
            nodes = self.semantic_splitter.get_nodes_from_documents([LlamaDocument(text=content)])
            chunks_text = [node.get_content() for node in nodes]
            metadata["content_type"] = "document"

        chunks = [{"text": chunk, "metadata": metadata} for chunk in chunks_text if chunk.strip()]
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
                vectors_config={"dense": qm.VectorParams(size=self.config.embedding_dimension, distance=qm.Distance.COSINE)},
                sparse_vectors_config={
                    "splade": qm.SparseVectorParams(),
                    "bm25": qm.SparseVectorParams(modifier=qm.Modifier.IDF)
                },
                hnsw_config=qm.HnswConfigDiff(m=48, ef_construct=1024),
                quantization_config=qm.ScalarQuantization(
                    scalar=qm.ScalarQuantizationConfig(
                        type=qm.ScalarType.INT8,
                        quantile=0.99,
                        always_ram=True
                    )
                )
            )
            self.client.create_payload_index(collection_name=self.config.collection_name, field_name="text", field_schema=qm.TextIndexParams(
                type="text", tokenizer=qm.TokenizerType.MULTILINGUAL, min_token_len=2, max_token_len=20, lowercase=True
            ))
            self.client.create_payload_index(collection_name=self.config.collection_name, field_name="metadata.source", field_schema=qm.PayloadSchemaType.KEYWORD)
            self.client.create_payload_index(collection_name=self.config.collection_name, field_name="metadata.content_type", field_schema=qm.PayloadSchemaType.KEYWORD)

    def upsert(self, chunks: List[Dict], dense_vectors: List[List[float]], sparse_vectors: Dict[str, List[Dict]]):
        points = [
            qm.PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": dense, 
                    "splade": qm.SparseVector(**sparse_vectors['splade'][i]),
                    "bm25": qm.SparseVector(**sparse_vectors['bm25'][i])
                },
                payload={**chunk['metadata'], "text": chunk['text']}
            ) for i, (chunk, dense) in enumerate(zip(chunks, dense_vectors))
        ]
        self.client.upsert(collection_name=self.config.collection_name, points=points, wait=True)

    def search(self, dense_query: List[float], sparse_queries: Dict, limit: int) -> List[Dict]:
        prefetch_queries = [
            qm.Prefetch(query=dense_query, using="dense", limit=limit * 5),
            qm.Prefetch(query=qm.SparseVector(**sparse_queries['splade']), using="splade", limit=limit * 5),
            qm.Prefetch(query=qm.SparseVector(**sparse_queries['bm25']), using="bm25", limit=limit * 5)
        ]
        hits = self.client.query_points(
            collection_name=self.config.collection_name,
            prefetch=prefetch_queries,
            query=qm.FusionQuery(fusion=qm.Fusion.RRF),
            limit=limit,
            with_payload=True,
            search_params=qm.SearchParams(quantization=qm.QuantizationSearchParams(rescore=True))
        )
        return [{"score": hit.score, **hit.payload} for hit in hits.points]

class MaxPerformancePipeline:
    """Orchestrates the entire ingestion and search pipeline."""
    def __init__(self):
        self.config = MaxPerformanceConfig()
        self.resource_manager = SystemResourceManager()
        self.cpu_executor = ThreadPoolExecutor(max_workers=self.config.cpu_threads)
        self.embedder = Embedder(self.config, self.cpu_executor)
        self.processor = DocumentProcessor(self.embedder)
        self.store = QdrantStore(self.config)
        self.reranker = CrossEncoder(self.config.reranker_model, device='cuda' if self.config.enable_gpu else 'cpu') if self.config.reranker_model else None

    async def ingest(self, source_path: str):
        metrics = PerformanceMetrics(start_time=time.time())
        source = Path(source_path)
        files = [p for p in source.rglob("*") if p.is_file()]
        metrics.total_files = len(files)

        all_chunks = []
        with ProcessPoolExecutor(max_workers=self.config.cpu_threads) as executor:
            futures = [executor.submit(self.processor.process_file, f, self.config.unstructured_strategy) for f in files]
            for i, future in enumerate(as_completed(futures)):
                all_chunks.extend(future.result())
                metrics.processed_files += 1
                logger.info(f"Processed file {i+1}/{len(files)}")
        
        metrics.initial_chunks = len(all_chunks)
        texts = [c['text'] for c in all_chunks]

        dense_vectors = await self.embedder.embed_dense_documents(texts)
        sparse_vectors = self.embedder.embed_sparse(texts)
        metrics.dense_embeddings = len(dense_vectors)
        metrics.sparse_splade_embeddings = len(sparse_vectors['splade'])
        metrics.sparse_bm25_embeddings = len(sparse_vectors['bm25'])

        self.store.upsert(all_chunks, dense_vectors, sparse_vectors)
        
        metrics.end_time = time.time()
        logger.info(f"Ingestion complete. Metrics: {json.dumps(metrics.summary(), indent=2)}")

    def search(self, query: str, limit: int = 10) -> List[Dict]:
        dense_query = self.embedder.embed_dense_query(query)
        # run sparse once and reuse outputs
        _sparse = self.embedder.embed_sparse([query])
        sparse_queries = {
            "splade": _sparse['splade'][0],
            "bm25": _sparse['bm25'][0]
        }
        
        results = self.store.search(dense_query, sparse_queries, limit * 5)
        
        if self.reranker:
            pairs = [[query, res['text']] for res in results]
            scores = self.reranker.predict(pairs)
            for res, score in zip(results, scores): res['rerank_score'] = float(score)
            results.sort(key=lambda x: x['rerank_score'], reverse=True)

        return results[:limit]

    def close(self):
        self.resource_manager.close()
        self.cpu_executor.shutdown(wait=True)

class QdrantEvaluator:
    """Measures the recall impact of quantization."""
    def __init__(self, pipeline: MaxPerformancePipeline):
        self.pipeline = pipeline
        self.config = pipeline.config
        self.client = QdrantClient(url=self.config.qdrant_url)
        self.non_quantized_collection = f"{self.config.collection_name}_no_quant"
        self.quantized_collection = f"{self.config.collection_name}_quantized"

    def setup_collections(self):
        # Non-quantized
        self.client.recreate_collection(
            collection_name=self.non_quantized_collection,
            vectors_config={"dense": qm.VectorParams(size=self.config.embedding_dimension, distance=qm.Distance.COSINE)},
            hnsw_config=qm.HnswConfigDiff(m=16, ef_construct=128)
        )
        # Quantized
        self.client.recreate_collection(
            collection_name=self.quantized_collection,
            vectors_config={"dense": qm.VectorParams(size=self.config.embedding_dimension, distance=qm.Distance.COSINE)},
            hnsw_config=qm.HnswConfigDiff(m=16, ef_construct=128),
            quantization_config=qm.ScalarQuantization(scalar=qm.ScalarQuantizationConfig(type=qm.ScalarType.INT8, quantile=0.99, always_ram=True))
        )

    async def run_benchmark(self, source_path: str, queries: List[str], k: int = 10):
        logger.info("Setting up collections for quantization benchmark...")
        self.setup_collections()

        logger.info(f"Ingesting sample data from {source_path}...")
        source = Path(source_path)
        files = [p for p in source.rglob("*") if p.is_file()][:20] # Limit to a sample
        
        all_chunks = []
        for f in files:
            all_chunks.extend(self.pipeline.processor.process_file(f, "fast"))
        
        texts = [c['text'] for c in all_chunks]
        dense_vectors = await self.pipeline.embedder.embed_dense_documents(texts)
        
        points = [qm.PointStruct(id=i, vector={"dense": vec}, payload=chunk) for i, (chunk, vec) in enumerate(zip(all_chunks, dense_vectors))]
        
        self.client.upsert(collection_name=self.non_quantized_collection, points=points, wait=True)
        self.client.upsert(collection_name=self.quantized_collection, points=points, wait=True)
        logger.info(f"Ingested {len(points)} sample points into both collections.")

        total_recall = 0
        for query in queries:
            query_vector = self.pipeline.embedder.embed_dense_query(query)
            
            # Ground truth (exact search on non-quantized)
            ground_truth_res = self.client.search(collection_name=self.non_quantized_collection, query_vector=query_vector, limit=k, search_params=qm.SearchParams(exact=True))
            ground_truth_ids = {hit.id for hit in ground_truth_res}

            # Approximate search on quantized collection
            quantized_res = self.client.search(collection_name=self.quantized_collection, query_vector=query_vector, limit=k, search_params=qm.SearchParams(quantization=qm.QuantizationSearchParams(rescore=True)))
            quantized_ids = {hit.id for hit in quantized_res}

            recall = len(ground_truth_ids.intersection(quantized_ids)) / k
            total_recall += recall
            logger.info(f"Query: '{query[:30]}...' -> Recall@{k}: {recall:.2f}")

        avg_recall = total_recall / len(queries)
        logger.info("-" * 50)
        logger.info(f"Average Recall@{k} for INT8 Quantization: {avg_recall:.4f}")
        logger.info("-" * 50)

def main_cli():
    """Command-line interface for the pipeline."""
    parser = argparse.ArgumentParser(description="High-Performance RAG Pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents from a source directory.")
    ingest_parser.add_argument("--source", type=str, required=True, help="Path to the source directory.")

    search_parser = subparsers.add_parser("search", help="Search the indexed documents.")
    search_parser.add_argument("--query", type=str, required=True, help="The search query.")
    search_parser.add_argument("--limit", type=int, default=10, help="Number of results to return.")

    eval_parser = subparsers.add_parser("evaluate-quantization", help="Benchmark recall of INT8 quantization.")
    eval_parser.add_argument("--source", type=str, required=True, help="Path to a small sample of source documents.")
    eval_parser.add_argument("--queries-file", type=str, required=True, help="JSON file with a list of test queries.")

    args = parser.parse_args()
    pipeline = MaxPerformancePipeline()

    try:
        if args.command == "ingest":
            asyncio.run(pipeline.ingest(args.source))
        elif args.command == "search":
            results = pipeline.search(args.query, args.limit)
            print(json.dumps(results, indent=2))
        elif args.command == "evaluate-quantization":
            with open(args.queries_file, 'r') as f:
                queries = [item['query'] for item in json.load(f)]
            evaluator = QdrantEvaluator(pipeline)
            asyncio.run(evaluator.run_benchmark(args.source, queries))

    finally:
        pipeline.close()

if __name__ == "__main__":
    main_cli()
