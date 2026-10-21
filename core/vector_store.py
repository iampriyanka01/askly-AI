"""
core/vector_store.py  -  DENSE RETRIEVAL (embeddings in FAISS or Pinecone)
==========================================================================

Role in the pipeline
    Stores one embedding per knowledge-base chunk and finds the chunks whose
    MEANING is closest to a question ("I can't get into my account" ~ "password
    reset"), which keyword search alone would miss.

        chunk text ─> embedding model (Gemini / OpenAI) ─> vector ─> FAISS | Pinecone
        question   ─> embed_query()                    ─> vector ─> top_k nearest chunks

Backends (VECTOR_BACKEND in .env)
    faiss     (default) local index in data/index/faiss - no extra account needed
    pinecone  managed serverless index; chunks live in one namespace

Embed-once guarantee (lesson from HireFlow: embedding costs money and quota)
    A manifest {chunk_id: content_hash} records what is already embedded.
    sync(chunks) only embeds chunks that are new or whose text changed, and deletes
    vectors for chunks that disappeared. Restarting the app embeds nothing.

Every public method returns a safe value instead of raising, so Askly keeps
working (BM25-only) when embeddings or the vector DB are unavailable. Check
is_ready() / status_message.

Public API
    initialize() -> bool
    sync(chunks) -> {'added', 'deleted', 'unchanged'}
    search(query, top_k, filters=None) -> [{'chunk_id','page_content','metadata','score'}]
    get_stats() / is_ready()
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from langchain_core.documents import Document
from utils.config import (VECTOR_BACKEND, FAISS_INDEX_DIR, PINECONE_API_KEY, PINECONE_INDEX_NAME,
                          PINECONE_NAMESPACE, PINECONE_METRIC, EMBEDDING_DIMENSION, PROJECT_ROOT)
from utils.utils import get_logger, get_embeddings, is_quota_error, resolve_provider

logger = get_logger(__name__)

EMBED_BATCH = 64


def _matches(metadata: Dict[str, Any], filters: Optional[Dict[str, Any]]) -> bool:
    """Same filter semantics for every backend: value -> equality, list -> membership."""
    if not filters:
        return True
    for key, cond in filters.items():
        val = metadata.get(key)
        if isinstance(cond, (list, tuple, set)):
            if val not in cond:
                return False
        elif val != cond:
            return False
    return True


class VectorStore:
    """Dense retrieval over knowledge-base chunks, backed by FAISS or Pinecone."""

    def __init__(self, backend: str = None, embeddings=None, index_dir: str = FAISS_INDEX_DIR):
        # Read the module setting at call time (the UI can switch to Pinecone at runtime).
        self.backend = (backend or globals()["VECTOR_BACKEND"] or "faiss").lower()
        self.embeddings = embeddings          # injectable (tests use fake embeddings)
        self.index_dir = Path(index_dir)
        self.faiss = None                     # langchain FAISS store
        self.pc_index = None                  # Pinecone index handle
        self.manifest: Dict[str, str] = {}    # chunk_id -> content_hash already embedded
        self._ready = False
        self.status_message = "not initialised"

    # ------------------------------------------------------------------ setup

    @property
    def _manifest_path(self) -> Path:
        if self.backend == "faiss":
            return self.index_dir / "manifest.json"
        return PROJECT_ROOT / "data" / "index" / f"pinecone_{PINECONE_INDEX_NAME}_{PINECONE_NAMESPACE}.json"

    def _embedding_signature(self) -> str:
        model = getattr(self.embeddings, "model", None) or type(self.embeddings).__name__
        return f"{resolve_provider()}:{model}:{EMBEDDING_DIMENSION}"

    def initialize(self) -> bool:
        """Load embeddings and connect to / load the index. Returns readiness."""
        try:
            if self.embeddings is None:
                self.embeddings = get_embeddings()
            if self.embeddings is None:
                self.status_message = "no embedding model (set GOOGLE_API_KEY or OPENAI_API_KEY)"
                logger.warning(f"Dense retrieval disabled: {self.status_message}")
                return False
            ok = self._init_faiss() if self.backend == "faiss" else self._init_pinecone()
            self._ready = ok
            return ok
        except Exception as e:
            self.status_message = f"initialisation failed: {e}"
            logger.error(f"Vector store {self.status_message}")
            self._ready = False
            return False

    def _load_manifest(self) -> Dict[str, Any]:
        try:
            return json.loads(self._manifest_path.read_text())
        except Exception:
            return {}

    def _save_manifest(self) -> None:
        self._manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self._manifest_path.write_text(json.dumps(
            {"embedding": self._embedding_signature(), "chunks": self.manifest}, indent=1))

    def _init_faiss(self) -> bool:
        try:
            import faiss  # noqa: F401  (fail early with a clear message)
        except ImportError:
            self.status_message = "faiss-cpu is not installed (pip install faiss-cpu)"
            logger.error(self.status_message)
            return False
        saved = self._load_manifest()
        if (self.index_dir / "index.faiss").exists() and saved.get("embedding") == self._embedding_signature():
            from langchain_community.vectorstores import FAISS
            self.faiss = FAISS.load_local(str(self.index_dir), self.embeddings,
                                          allow_dangerous_deserialization=True,  # our own file
                                          normalize_L2=True)
            self.manifest = saved.get("chunks", {})
            self.status_message = f"FAISS loaded ({len(self.manifest)} chunks)"
        else:
            if saved:
                logger.warning("Embedding model changed since the FAISS index was built - rebuilding it.")
            self.faiss, self.manifest = None, {}
            self.status_message = "FAISS ready (empty - will build on first sync)"
        logger.info(self.status_message)
        return True

    def _init_pinecone(self) -> bool:
        if not PINECONE_API_KEY:
            self.status_message = "PINECONE_API_KEY not set"
            logger.warning(self.status_message)
            return False
        from pinecone import Pinecone, ServerlessSpec
        client = Pinecone(api_key=PINECONE_API_KEY)
        names = [i.name for i in client.list_indexes()]
        if PINECONE_INDEX_NAME in names:
            dim = client.describe_index(PINECONE_INDEX_NAME).dimension
            if dim != EMBEDDING_DIMENSION:
                # Never delete a production index automatically (HireFlow HW5 lesson).
                self.status_message = (f"Pinecone index '{PINECONE_INDEX_NAME}' has dimension {dim} but "
                                       f"EMBEDDING_DIMENSION={EMBEDDING_DIMENSION}. Fix the setting or use a new index name.")
                logger.error(self.status_message)
                return False
        else:
            client.create_index(name=PINECONE_INDEX_NAME, dimension=EMBEDDING_DIMENSION,
                                metric=PINECONE_METRIC, spec=ServerlessSpec(cloud="aws", region="us-east-1"))
            for _ in range(30):
                status = client.describe_index(PINECONE_INDEX_NAME).status
                if (status.get("ready") if isinstance(status, dict) else getattr(status, "ready", False)):
                    break
                time.sleep(2)
        self.pc_index = client.Index(PINECONE_INDEX_NAME)
        saved = self._load_manifest()
        self.manifest = saved.get("chunks", {}) if saved.get("embedding") == self._embedding_signature() else {}
        self.status_message = f"Pinecone '{PINECONE_INDEX_NAME}/{PINECONE_NAMESPACE}' connected"
        logger.info(self.status_message)
        return True

    def is_ready(self) -> bool:
        return self._ready

    # ------------------------------------------------------------------ writes

    def sync(self, chunks: List[Document]) -> Dict[str, int]:
        """Make the index match `chunks`: embed new/changed chunks, delete removed ones."""
        result = {"added": 0, "deleted": 0, "unchanged": 0}
        if not self._ready:
            return result
        wanted = {c.metadata["chunk_id"]: c.metadata["content_hash"] for c in chunks}
        to_add = [c for c in chunks if self.manifest.get(c.metadata["chunk_id"]) != c.metadata["content_hash"]]
        to_delete = [cid for cid, h in self.manifest.items() if wanted.get(cid) != h]
        result["unchanged"] = len(chunks) - len(to_add)
        try:
            if to_delete:
                self._delete(to_delete)
                for cid in to_delete:
                    self.manifest.pop(cid, None)
                result["deleted"] = len(to_delete)
            for start in range(0, len(to_add), EMBED_BATCH):
                batch = to_add[start:start + EMBED_BATCH]
                vectors = self.embeddings.embed_documents([c.page_content for c in batch])
                self._upsert(batch, vectors)
                for c in batch:
                    self.manifest[c.metadata["chunk_id"]] = c.metadata["content_hash"]
                result["added"] += len(batch)
                self._persist()
        except Exception as e:
            if is_quota_error(e):
                logger.warning(f"Embedding quota hit after {result['added']} chunks - the rest will be "
                               "embedded on the next sync.")
            else:
                logger.error(f"Vector sync failed: {e}")
        finally:
            self._persist()
        if result["added"] or result["deleted"]:
            logger.info(f"Vector sync: {result}")
        return result

    def add_chunks(self, chunks: List[Document]) -> Dict[str, int]:
        """Embed and add chunks without deleting anything else (used for uploads)."""
        if not self._ready:
            return {"added": 0, "deleted": 0, "unchanged": 0}
        keep = [c for c in chunks if self.manifest.get(c.metadata["chunk_id"]) != c.metadata["content_hash"]]
        old_ids = [c.metadata["chunk_id"] for c in keep if c.metadata["chunk_id"] in self.manifest]
        if old_ids:
            self._delete(old_ids)
            for cid in old_ids:
                self.manifest.pop(cid, None)
        added = 0
        try:
            for start in range(0, len(keep), EMBED_BATCH):
                batch = keep[start:start + EMBED_BATCH]
                self._upsert(batch, self.embeddings.embed_documents([c.page_content for c in batch]))
                for c in batch:
                    self.manifest[c.metadata["chunk_id"]] = c.metadata["content_hash"]
                added += len(batch)
        except Exception as e:
            logger.error(f"Adding chunks failed: {e}")
        self._persist()
        return {"added": added, "deleted": len(old_ids), "unchanged": len(chunks) - len(keep)}

    def _upsert(self, chunks: List[Document], vectors: List[List[float]]) -> None:
        ids = [c.metadata["chunk_id"] for c in chunks]
        if self.backend == "faiss":
            from langchain_community.vectorstores import FAISS
            pairs = list(zip([c.page_content for c in chunks], vectors))
            metas = [dict(c.metadata) for c in chunks]
            if self.faiss is None:
                self.faiss = FAISS.from_embeddings(pairs, self.embeddings, metadatas=metas, ids=ids,
                                                   normalize_L2=True)
            else:
                self.faiss.add_embeddings(pairs, metadatas=metas, ids=ids)
        else:
            payload = []
            for c, v, cid in zip(chunks, vectors, ids):
                meta = {k: v2 for k, v2 in c.metadata.items() if v2 not in (None, "")}
                meta["page_content"] = c.page_content
                payload.append({"id": cid, "values": v, "metadata": meta})
            for start in range(0, len(payload), 100):
                self.pc_index.upsert(vectors=payload[start:start + 100], namespace=PINECONE_NAMESPACE)

    def _delete(self, ids: List[str]) -> None:
        if self.backend == "faiss":
            if self.faiss is not None:
                present = [i for i in ids if i in self.faiss.index_to_docstore_id.values()]
                if present:
                    self.faiss.delete(present)
        else:
            self.pc_index.delete(ids=ids, namespace=PINECONE_NAMESPACE)

    def _persist(self) -> None:
        if self.backend == "faiss" and self.faiss is not None:
            self.index_dir.mkdir(parents=True, exist_ok=True)
            self.faiss.save_local(str(self.index_dir))
        self._save_manifest()

    # ------------------------------------------------------------------ reads

    def search(self, query: str, top_k: int = 10, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Semantic search. 'score' is cosine similarity (higher = closer)."""
        if not self._ready or not query.strip():
            return []
        try:
            qvec = self.embeddings.embed_query(query)   # query-side task type (HireFlow HW2)
            if self.backend == "faiss":
                if self.faiss is None:
                    return []
                filt: Optional[Callable] = (lambda md: _matches(md, filters)) if filters else None
                hits = self.faiss.similarity_search_with_score_by_vector(
                    qvec, k=top_k, filter=filt, fetch_k=max(top_k * 5, 50))
                # normalize_L2=True + IndexFlatL2 -> squared distance d = 2 - 2cos
                return [{"chunk_id": d.metadata.get("chunk_id"), "page_content": d.page_content,
                         "metadata": d.metadata, "score": float(1 - dist / 2)} for d, dist in hits]
            pc_filter = None
            if filters:
                pc_filter = {k: ({"$in": list(v)} if isinstance(v, (list, tuple, set)) else {"$eq": v})
                             for k, v in filters.items()}
            res = self.pc_index.query(vector=qvec, top_k=top_k, namespace=PINECONE_NAMESPACE,
                                      include_metadata=True, filter=pc_filter)
            out = []
            for m in (res.matches or []):
                md = dict(m.metadata or {})
                text = md.pop("page_content", "")
                out.append({"chunk_id": m.id, "page_content": text, "metadata": md, "score": float(m.score)})
            return out
        except Exception as e:
            if is_quota_error(e):
                logger.warning("Embedding quota exceeded - dense search skipped for this query")
            else:
                logger.error(f"Dense search failed: {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        stats = {"backend": self.backend, "ready": self._ready, "status": self.status_message,
                 "embedded_chunks": len(self.manifest)}
        try:
            if self._ready and self.backend == "pinecone":
                s = self.pc_index.describe_index_stats()
                stats["total_vector_count"] = getattr(s, "total_vector_count", None)
            elif self._ready and self.faiss is not None:
                stats["total_vector_count"] = self.faiss.index.ntotal
        except Exception as e:
            stats["stats_error"] = str(e)
        return stats
