"""Модели для RAG: эмбеддинги (deepvk/USER-bge-m3) и реранкер (BAAI/bge-reranker-v2-m3).

Загружаются один раз при первом обращении и живут до конца процесса. На видеокарте —
в половинной точности (fp16): вдвое меньше памяти, качество практически то же.
Названия моделей можно сменить в .env: RAG_EMBED_MODEL, RAG_RERANK_MODEL.
"""
from __future__ import annotations

import os
import threading
import time

EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "deepvk/USER-bge-m3")
RERANK_MODEL = os.getenv("RAG_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
EMBEDDING_DIM = 1024
MAX_SEQ_LENGTH = 512   # фрагменты короче, а длинный контекст съедает видеопамять

_lock = threading.Lock()
_embedder = None
_reranker = None


def _device() -> str:
    forced = os.getenv("RAG_DEVICE")
    if forced:
        return forced
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _require_sentence_transformers():
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "Не установлен пакет sentence-transformers. Выполните: "
            "venv\\Scripts\\python.exe -m pip install sentence-transformers") from None


def get_embedder():
    global _embedder
    with _lock:
        if _embedder is None:
            _require_sentence_transformers()
            from sentence_transformers import SentenceTransformer

            device = _device()
            started = time.time()
            model = SentenceTransformer(EMBED_MODEL, device=device)
            model.max_seq_length = MAX_SEQ_LENGTH
            if device == "cuda":
                model.half()
            print(f"[RAG] эмбеддинги {EMBED_MODEL} загружены на {device} за {time.time() - started:.0f} с")
            _embedder = model
    return _embedder


def get_reranker():
    global _reranker
    with _lock:
        if _reranker is None:
            _require_sentence_transformers()
            from sentence_transformers import CrossEncoder

            device = _device()
            started = time.time()
            model = CrossEncoder(RERANK_MODEL, device=device, max_length=MAX_SEQ_LENGTH)
            if device == "cuda":
                model.model.half()
            print(f"[RAG] реранкер {RERANK_MODEL} загружен на {device} за {time.time() - started:.0f} с")
            _reranker = model
    return _reranker


def embed(texts: list[str], batch_size: int = 16, show_progress: bool = False) -> list[list[float]]:
    """Векторы длины 1024, нормированные (косинусная близость = скалярное произведение)."""
    model = get_embedder()
    vectors = model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                           convert_to_numpy=True, show_progress_bar=show_progress)
    return [v.astype(float).tolist() for v in vectors]


def rerank_scores(query: str, texts: list[str], batch_size: int = 8) -> list[float]:
    """Оценка релевантности пары «вопрос — фрагмент»: чем больше, тем лучше."""
    if not texts:
        return []
    model = get_reranker()
    scores = model.predict([(query, t) for t in texts], batch_size=batch_size, show_progress_bar=False)
    return [float(s) for s in scores]
