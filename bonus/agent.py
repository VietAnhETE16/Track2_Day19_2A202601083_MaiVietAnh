"""HybridMemoryAgent — Personal AI Assistant Memory POC for Vietnamese Users.

Combines:
  1. Episodic Memory (Qdrant Vector Store) with per-user isolation filters.
  2. Stable Profile + Recent Velocity (Feast Online Feature Store).
  3. Prompt Context Assembler for LLM grounding and personalization.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

COLLECTION_NAME = "agent_memories"
VECTOR_DIM = 384


class HybridMemoryAgent:
    """Agent that manages dual-tier memory (Episodic Vector Store + Feast Profile Store)."""

    def __init__(
        self,
        feast_repo_path: str | Path | None = None,
        embedder: TextEmbedding | None = None,
        qdrant_client: QdrantClient | None = None,
    ) -> None:
        self.embedder = embedder or TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
        self.client = qdrant_client or QdrantClient(":memory:")

        # Initialize Qdrant collection if not exists
        existing = {c.name for c in self.client.get_collections().collections}
        if COLLECTION_NAME not in existing:
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
            )

        self._point_id = 0
        self.fs = None
        if feast_repo_path:
            repo_p = Path(feast_repo_path)
            if (repo_p / "registry.db").exists() or (repo_p / "feature_store.yaml").exists():
                try:
                    from feast import FeatureStore
                    self.fs = FeatureStore(repo_path=str(repo_p))
                except Exception:
                    self.fs = None

    def remember(self, text: str, user_id: str = "u_001", metadata: dict[str, Any] | None = None) -> list[int]:
        """Chunk text, compute embeddings, and store with strict user_id isolation."""
        if not text or not text.strip():
            return []

        # Simple semantic-friendly chunking (by paragraphs or sentences)
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
        if not paragraphs:
            paragraphs = [text.strip()]

        points: list[PointStruct] = []
        point_ids: list[int] = []
        vectors = list(self.embedder.embed(paragraphs))

        for chunk_text, vec in zip(paragraphs, vectors):
            pid = self._point_id
            self._point_id += 1
            point_ids.append(pid)

            payload = {
                "user_id": user_id,
                "text": chunk_text,
                "created_at": time.time(),
            }
            if metadata:
                payload.update(metadata)

            points.append(PointStruct(id=pid, vector=vec.tolist(), payload=payload))

        self.client.upsert(collection_name=COLLECTION_NAME, points=points)
        return point_ids

    def recall(self, query: str, user_id: str = "u_001", top_k: int = 3) -> str:
        """Retrieve top-K memories isolated by user_id + fetch Feast profile -> assemble context."""
        # 1. Fetch user profile from Feast online store
        profile_info: dict[str, Any] = {
            "reading_speed_wpm": 200,
            "preferred_language": "vi",
            "topic_affinity": "ai_ml",
            "queries_last_hour": 5,
            "distinct_topics_24h": 3,
        }

        if self.fs is not None:
            try:
                features = self.fs.get_online_features(
                    features=[
                        "user_profile_features:reading_speed_wpm",
                        "user_profile_features:preferred_language",
                        "user_profile_features:topic_affinity",
                        "query_velocity_features:queries_last_hour",
                        "query_velocity_features:distinct_topics_24h",
                    ],
                    entity_rows=[{"user_id": user_id}],
                ).to_dict()

                for k, v in features.items():
                    clean_k = k.split(":")[-1]
                    if v and v[0] is not None:
                        profile_info[clean_k] = v[0]
            except Exception:
                pass  # fallback to standard profile defaults

        # 2. Vector search on episodic memory with strict user_id filtering (OWASP / Privacy)
        q_vec = next(self.embedder.embed([query])).tolist()
        user_filter = Filter(
            must=[
                FieldCondition(key="user_id", match=MatchValue(value=user_id))
            ]
        )

        res = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=q_vec,
            query_filter=user_filter,
            limit=top_k,
        )

        memory_hits = [p.payload["text"] for p in res.points if p.payload and "text" in p.payload]

        # 3. Assemble complete prompt context
        context_lines = [
            "=== SYSTEM CONTEXT (HYBRID AI MEMORY) ===",
            f"User ID: {user_id}",
            f"Profile: Language={profile_info.get('preferred_language', 'vi')}, "
            f"Speed={profile_info.get('reading_speed_wpm', 200)} wpm, "
            f"Affinity={profile_info.get('topic_affinity', 'general')}",
            f"Recent Activity: {profile_info.get('queries_last_hour', 0)} queries in last hour, "
            f"{profile_info.get('distinct_topics_24h', 1)} distinct topics in 24h",
            "",
            "=== RECALLED EPISODIC MEMORIES ===",
        ]

        if memory_hits:
            for idx, mem in enumerate(memory_hits, start=1):
                context_lines.append(f"{idx}. {mem}")
        else:
            context_lines.append("(No relevant past episodic memories found for this user.)")

        context_lines.append("==========================================")
        return "\n".join(context_lines)
