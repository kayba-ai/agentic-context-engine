"""Similarity detection for skill deduplication."""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING, List, Optional, Tuple

from ..protocols.deduplication import DeduplicationConfig

if TYPE_CHECKING:
    from ..core.skillbook import Skill
    from ..core.skillbook import Skillbook

logger = logging.getLogger(__name__)


def _has(module: str) -> bool:
    """Return True if *module* can be imported."""
    try:
        importlib.import_module(module)
        return True
    except ImportError:
        return False


class SimilarityDetector:
    """Detect similar skill pairs using cosine similarity on embeddings."""

    def __init__(self, config: DeduplicationConfig | None = None) -> None:
        self.config = config or DeduplicationConfig()
        self._model: object | None = None  # lazy sentence-transformers model

    # ------------------------------------------------------------------
    # Single / batch embedding computation
    # ------------------------------------------------------------------

    def compute_embedding(self, text: str) -> Optional[List[float]]:
        """Compute embedding for a single text."""
        if self.config.embedding_provider == "litellm":
            return self._embed_litellm(text)
        return self._embed_st(text)

    def compute_embeddings_batch(
        self, texts: List[str]
    ) -> List[Optional[List[float]]]:
        """Compute embeddings for multiple texts (more efficient)."""
        if not texts:
            return []
        if self.config.embedding_provider == "litellm":
            return self._embed_batch_litellm(texts)
        return self._embed_batch_st(texts)

    # ------------------------------------------------------------------
    # LiteLLM provider
    # ------------------------------------------------------------------

    def _embed_litellm(self, text: str) -> Optional[List[float]]:
        if not _has("litellm"):
            logger.warning("LiteLLM not available for embeddings")
            return None
        try:
            import litellm

            response = litellm.embedding(
                model=self.config.embedding_model, input=[text]
            )
            return response.data[0]["embedding"]
        except Exception as e:
            logger.warning("Failed to compute embedding via LiteLLM: %s", e)
            return None

    def _embed_batch_litellm(
        self, texts: List[str]
    ) -> List[Optional[List[float]]]:
        if not _has("litellm"):
            logger.warning("LiteLLM not available for embeddings")
            return [None] * len(texts)
        try:
            import litellm

            response = litellm.embedding(
                model=self.config.embedding_model, input=texts
            )
            return [item["embedding"] for item in response.data]
        except Exception as e:
            logger.warning(
                "Failed to compute batch embeddings via LiteLLM: %s", e
            )
            return [None] * len(texts)

    # ------------------------------------------------------------------
    # sentence-transformers provider
    # ------------------------------------------------------------------

    def _embed_st(self, text: str) -> Optional[List[float]]:
        if not _has("sentence_transformers"):
            logger.warning(
                "sentence-transformers not available for embeddings"
            )
            return None
        try:
            model = self._get_st_model()
            embedding = model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            logger.warning(
                "Failed to compute embedding via sentence-transformers: %s", e
            )
            return None

    def _embed_batch_st(
        self, texts: List[str]
    ) -> List[Optional[List[float]]]:
        if not _has("sentence_transformers"):
            logger.warning(
                "sentence-transformers not available for embeddings"
            )
            return [None] * len(texts)
        try:
            model = self._get_st_model()
            embeddings = model.encode(texts, convert_to_numpy=True)
            return [emb.tolist() for emb in embeddings]
        except Exception as e:
            logger.warning(
                "Failed to compute batch embeddings via "
                "sentence-transformers: %s",
                e,
            )
            return [None] * len(texts)

    def _get_st_model(self):
        """Lazy-load the sentence-transformers model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.config.local_model_name)
        return self._model

    # ------------------------------------------------------------------
    # Cosine similarity
    # ------------------------------------------------------------------

    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two embedding vectors."""
        if not _has("numpy"):
            # Pure-Python fallback
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = sum(x * x for x in a) ** 0.5
            norm_b = sum(x * x for x in b) ** 0.5
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return dot / (norm_a * norm_b)

        import numpy as np

        a_arr = np.array(a)
        b_arr = np.array(b)
        dot = np.dot(a_arr, b_arr)
        norm_a = np.linalg.norm(a_arr)
        norm_b = np.linalg.norm(b_arr)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    # ------------------------------------------------------------------
    # High-level API
    # ------------------------------------------------------------------

    def ensure_embeddings(self, skillbook: "Skillbook") -> int:
        """Ensure all active skills have embeddings computed.

        Returns:
            Number of new embeddings computed.
        """
        needs = [s for s in skillbook.skills() if s.embedding is None]
        if not needs:
            return 0

        texts = [s.content for s in needs]
        embeddings = self.compute_embeddings_batch(texts)

        count = 0
        for skill, embedding in zip(needs, embeddings):
            if embedding is not None:
                skill.embedding = embedding
                count += 1

        logger.info("Computed %d embeddings for skills", count)
        return count

    def detect_similar_pairs(
        self,
        skillbook: "Skillbook",
        threshold: float | None = None,
    ) -> List[Tuple["Skill", "Skill", float]]:
        """Find all skill pairs with similarity >= *threshold*.

        Returns:
            Sorted list of ``(skill_a, skill_b, similarity)`` tuples
            (descending by score).
        """
        threshold = threshold or self.config.similarity_threshold
        similar_pairs: List[Tuple["Skill", "Skill", float]] = []

        skills = skillbook.skills(include_invalid=False)

        if self.config.within_section_only:
            sections: dict[str, list] = {}
            for skill in skills:
                sections.setdefault(skill.section, []).append(skill)
            for section_skills in sections.values():
                similar_pairs.extend(
                    self._find_similar(section_skills, skillbook, threshold)
                )
        else:
            similar_pairs = self._find_similar(skills, skillbook, threshold)

        similar_pairs.sort(key=lambda x: x[2], reverse=True)
        return similar_pairs

    def _find_similar(
        self,
        skills: List["Skill"],
        skillbook: "Skillbook",
        threshold: float,
    ) -> List[Tuple["Skill", "Skill", float]]:
        pairs: List[Tuple["Skill", "Skill", float]] = []
        for i, skill_a in enumerate(skills):
            if skill_a.embedding is None:
                continue
            for skill_b in skills[i + 1 :]:
                if skill_b.embedding is None:
                    continue
                if skillbook.has_keep_decision(skill_a.id, skill_b.id):
                    continue
                sim = self.cosine_similarity(
                    skill_a.embedding, skill_b.embedding
                )
                if sim >= threshold:
                    pairs.append((skill_a, skill_b, sim))
        return pairs
