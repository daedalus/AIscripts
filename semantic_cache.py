# FIXED & HARDENED VERSION
# Key fixes applied in this pass:
# - _evict LRU trim: O(n²) while+list-scan → O(n) set-based
# - process: `if cached:` → `if cached is not None:` (falsy response guard)
# - safe_norm: silent zero-vector on NaN/zero → raises ValueError
# - CLI: restored --ttl flag and EMBED_BACKEND env var

import numpy as np
import faiss
import xxhash
import time
import os
import argparse
import threading
from collections import OrderedDict

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from transformers import AutoTokenizer, AutoModel
import torch

INTENT_THRESHOLD = 0.78
CONTEXT_THRESHOLD = 0.88
TTL_SECONDS = 3600
TOP_K = 8
EVICT_EVERY_N = 100
MAX_ENTRIES = 20_000
MAX_CTX_TOKENS = 512
# Sliding window stride; chunks overlap by (MAX_CTX_TOKENS - CTX_STRIDE) tokens.
CTX_STRIDE = 256


# -----------------------------
# Utils
# -----------------------------

def safe_norm(v: np.ndarray) -> np.ndarray:
    """
    L2-normalise v.  Raises ValueError on None, empty, non-finite, or zero-norm
    vectors rather than silently returning a zero vector — a zero vector stored
    in the FAISS index would produce score=0 for every query, causing subtle
    false-misses that are very hard to debug.
    """
    if v is None or len(v) == 0:
        raise ValueError("safe_norm: received None or empty vector")
    n = np.linalg.norm(v)
    if not np.isfinite(n) or n == 0:
        raise ValueError(f"safe_norm: non-finite or zero norm ({n})")
    return v / n


def dot(a: np.ndarray, b: np.ndarray) -> float:
    """
    Inner product of two L2-normalised vectors == cosine similarity.
    None guards are kept so callers don't need to check before calling.
    """
    if a is None or b is None:
        return 0.0
    return float(np.dot(a, b))


# -----------------------------
# Embedders
# -----------------------------

class LocalEmbedder:
    """E5-compatible embedder with query/passage prefix split and sliding-window
    chunking for contexts that exceed the model's token limit."""

    def __init__(self, model_name="intfloat/e5-base-v2", device=None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)

    def _encode(self, texts):
        enc = self.tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            out = self.model(**enc)
        emb = self._mean_pooling(out, enc["attention_mask"])
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb

    def embed_query(self, text: str) -> np.ndarray:
        return safe_norm(self._encode(["query: " + text])[0].cpu().numpy().astype("float32"))

    def embed_passage(self, text: str) -> np.ndarray:
        return safe_norm(self._encode(["passage: " + text])[0].cpu().numpy().astype("float32"))

    def embed_context_chunked(self, text: str) -> np.ndarray:
        toks = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(toks) <= MAX_CTX_TOKENS:
            return self.embed_passage(text)

        vecs = []
        for i in range(0, len(toks), CTX_STRIDE):
            chunk = toks[i : i + MAX_CTX_TOKENS]
            txt = self.tokenizer.decode(chunk, clean_up_tokenization_spaces=False)
            vecs.append(self.embed_passage(txt))

        v = np.mean(np.stack(vecs, axis=0), axis=0)
        return safe_norm(v)


class OpenAIEmbedder:
    def __init__(self, api_key=None, base_url=None, model="text-embedding-3-small"):
        if not OPENAI_AVAILABLE:
            raise RuntimeError("openai package not installed")
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
        )
        self.model = model

    def embed_query(self, text: str) -> np.ndarray:
        r = self.client.embeddings.create(model=self.model, input="query: " + text)
        return safe_norm(np.array(r.data[0].embedding, dtype=np.float32))

    def embed_context_chunked(self, text: str) -> np.ndarray:
        r = self.client.embeddings.create(model=self.model, input="passage: " + text)
        return safe_norm(np.array(r.data[0].embedding, dtype=np.float32))


# -----------------------------
# Canonicalization
# -----------------------------

def canonicalize(text: str) -> str:
    return " ".join(text.lower().strip().split())


def extract_intent(prompt: str) -> str:
    return canonicalize(prompt)


def context_hash(context: str) -> str:
    return xxhash.xxh64(context.encode()).hexdigest()


# -----------------------------
# Cache Entry
# -----------------------------

class CacheEntry:
    __slots__ = ("intent_vec", "ctx_hash", "ctx_vec", "response", "timestamp", "ttl", "hits")

    def __init__(self, intent_vec, ctx_hash, ctx_vec, response, ttl=TTL_SECONDS):
        self.intent_vec = intent_vec
        self.ctx_hash   = ctx_hash
        self.ctx_vec    = ctx_vec
        self.response   = response
        self.timestamp  = time.monotonic()
        self.ttl        = ttl
        self.hits       = 0

    def is_expired(self) -> bool:
        return (time.monotonic() - self.timestamp) > self.ttl


# -----------------------------
# Cache (thread-safe, O(1) LRU + TTL eviction)
# -----------------------------

class SemanticCache:
    """
    LRU order is tracked in `_lru`, an OrderedDict[id(entry) -> entry].
    - add/touch: O(1)
    - evict LRU trim: O(n) — iterates _lru front-to-back once to collect
      ids to drop, then filters with a set for O(1) membership tests.

    FAISS index positions correspond 1-to-1 with _entries indices.
    The index is rebuilt after every eviction; evictions are infrequent
    (every EVICT_EVERY_N inserts or when at capacity).
    """

    def __init__(self, dim: int, evict_every=EVICT_EVERY_N, max_entries=MAX_ENTRIES):
        self.dim              = dim
        self.evict_every      = evict_every
        self.max_entries      = max_entries
        self._lock            = threading.RLock()
        self._entries: list[CacheEntry] = []
        self._index           = faiss.IndexFlatIP(dim)
        self._ops_since_evict = 0
        self._lru: OrderedDict[int, CacheEntry] = OrderedDict()

    def add(self, entry: CacheEntry) -> None:
        with self._lock:
            self._ops_since_evict += 1
            if (
                self._ops_since_evict >= self.evict_every
                or len(self._entries) >= self.max_entries
            ):
                self._evict()
                self._ops_since_evict = 0

            self._entries.append(entry)
            self._lru[id(entry)] = entry
            self._index.add(np.array([entry.intent_vec], dtype="float32"))

    def search(
        self,
        intent_vec: np.ndarray,
        ctx_hash: str,
        ctx_vec: np.ndarray,
    ) -> str | None:
        with self._lock:
            if not self._entries:
                return None

            k = min(TOP_K, len(self._entries))
            D, I = self._index.search(np.array([intent_vec], dtype="float32"), k)

            for score, idx in zip(D[0], I[0]):
                if idx < 0 or score < INTENT_THRESHOLD:
                    break  # results are descending; nothing useful beyond here

                if idx >= len(self._entries):
                    continue  # stale index position after partial eviction

                e = self._entries[idx]
                if e.is_expired():
                    continue

                if e.ctx_hash == ctx_hash:
                    e.hits += 1
                    self._touch(e)
                    return e.response

                if score > (INTENT_THRESHOLD + 0.05):
                    if dot(e.ctx_vec, ctx_vec) > CONTEXT_THRESHOLD:
                        e.hits += 1
                        self._touch(e)
                        return e.response

            return None

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def _touch(self, entry: CacheEntry) -> None:
        """Mark entry as most-recently used. O(1)."""
        key = id(entry)
        if key in self._lru:
            self._lru.move_to_end(key)

    def _evict(self) -> None:
        """
        1. Drop TTL-expired entries.
        2. If still over capacity, evict the LRU entries (front of _lru)
           until within budget — O(n) total via set membership.
        3. Rebuild FAISS index from survivors.
        """
        # Step 1: TTL filter
        live_ids = {id(e) for e in self._entries if not e.is_expired()}

        # Step 2: LRU capacity trim — walk oldest→newest, collect ids to drop
        if len(live_ids) > self.max_entries:
            drop: set[int] = set()
            for eid in self._lru:  # OrderedDict iterates insertion/LRU order
                if len(live_ids) - len(drop) <= self.max_entries:
                    break
                if eid in live_ids:
                    drop.add(eid)
            live_ids -= drop

        # Rebuild _entries (preserving insertion order) and _lru (preserving LRU order)
        self._entries = [e for e in self._entries if id(e) in live_ids]
        self._lru     = OrderedDict(
            (eid, e) for eid, e in self._lru.items() if eid in live_ids
        )

        # Rebuild FAISS index
        self._index = faiss.IndexFlatIP(self.dim)
        if self._entries:
            vecs = np.array([e.intent_vec for e in self._entries], dtype="float32")
            self._index.add(vecs)


# -----------------------------
# Validation
# -----------------------------

def validate_response(prompt: str, response: str) -> bool:
    """
    Sanity-check a freshly generated LLM response before caching.
    Requires response to be non-empty (≥5 chars) and share at least 20%
    token overlap with the prompt.  Not applied to cache hits.
    """
    if not response or len(response) < 5:
        return False
    p = set(prompt.lower().split())
    r = set(response.lower().split())
    return len(p & r) >= max(1, len(p) // 5)


# -----------------------------
# System
# -----------------------------

class SemanticCacheSystem:
    def __init__(self, embedder, ttl: int = TTL_SECONDS):
        self.embedder = embedder
        self.ttl      = ttl

        test_vec = self.embedder.embed_query("test")
        self.dim  = len(test_vec)
        self.cache = SemanticCache(self.dim)

    def process(self, prompt: str, context: str) -> str:
        intent    = extract_intent(prompt)
        ctx_hash_ = context_hash(context)

        intent_vec = self.embedder.embed_query(intent)
        ctx_vec    = self.embedder.embed_context_chunked(context)

        cached = self.cache.search(intent_vec, ctx_hash_, ctx_vec)
        if cached is not None:  # explicit None check — empty string is a valid response
            print("[CACHE HIT]")
            return cached

        # --- Replace this stub with your actual LLM call ---
        response = f"[LLM] {prompt}"
        # ---------------------------------------------------

        if validate_response(prompt, response):
            self.cache.add(
                CacheEntry(intent_vec, ctx_hash_, ctx_vec, response, ttl=self.ttl)
            )

        print("[CACHE MISS]")
        return response


# -----------------------------
# CLI
# -----------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Semantic Cache demo")
    parser.add_argument(
        "--backend",
        choices=["local", "openai"],
        default=os.getenv("EMBED_BACKEND", "local"),
    )
    parser.add_argument("--openai-key",   default=os.getenv("OPENAI_API_KEY"))
    parser.add_argument("--openai-base",  default=os.getenv("OPENAI_BASE_URL"))
    parser.add_argument(
        "--openai-model",
        default=os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small"),
    )
    parser.add_argument(
        "--ttl",
        type=int,
        default=TTL_SECONDS,
        help="Cache entry TTL in seconds (default: %(default)s)",
    )

    args = parser.parse_args()

    if args.backend == "openai":
        embedder = OpenAIEmbedder(
            api_key=args.openai_key,
            base_url=args.openai_base,
            model=args.openai_model,
        )
    else:
        embedder = LocalEmbedder()

    system = SemanticCacheSystem(embedder, ttl=args.ttl)

    ctx = "def add(a, b): return a + b"

    queries = [
        "optimize this code",         # MISS  — seeds the cache
        "make it faster code",        # HIT   — semantically similar, same ctx
        "the code is slowe",          # HIT   — typo but close enough
        "optimize the code",          # HIT   — near-identical
        "fix the code",               # borderline — depends on threshold
        "the cat is under the table", # MISS  — unrelated
    ]

    for q in queries:
        result = system.process(q, ctx)
        print(f"  → {result}\n")