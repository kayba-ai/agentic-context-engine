"""Similarity detection for bullet deduplication."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional, Tuple

from ..features import has_litellm, has_numpy, has_sentence_transformers
from .config import DeduplicationConfig

if TYPE_CHECKING:
    from ..playbook import Bullet, Playbook

logger = logging.getLogger(__name__)


class SimilarityDetector:
    """Detect similar bullet pairs using cosine similarity on embeddings."""

    def __init__(self, config: Optional[DeduplicationConfig] = None):
        self.config = config or DeduplicationConfig()
        self._model = None  # Lazy load sentence-transformers model

    def compute_embedding(self, text: str) -> Optional[List[float]]:
        """Compute embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats, or None if embedding fails
        """
        if self.config.embedding_provider == "litellm":
            return self._compute_embedding_litellm(text)
        else:
            return self._compute_embedding_sentence_transformers(text)

    def compute_embeddings_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Compute embeddings for multiple texts (more efficient).

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors (None for any that fail)
        """
        if not texts:
            return []

        if self.config.embedding_provider == "litellm":
            return self._compute_embeddings_batch_litellm(texts)
        else:
            return self._compute_embeddings_batch_sentence_transformers(texts)

    def _compute_embedding_litellm(self, text: str) -> Optional[List[float]]:
        """Compute embedding using LiteLLM."""
        if not has_litellm():
            logger.warning("LiteLLM not available for embeddings")
            return None

        try:
            import litellm

            response = litellm.embedding(
                model=self.config.embedding_model,
                input=[text],
            )
            return response.data[0]["embedding"]
        except Exception as e:
            logger.warning(f"Failed to compute embedding via LiteLLM: {e}")
            return None

    def _compute_embeddings_batch_litellm(
        self, texts: List[str]
    ) -> List[Optional[List[float]]]:
        """Batch compute embeddings using LiteLLM."""
        if not has_litellm():
            logger.warning("LiteLLM not available for embeddings")
            return [None] * len(texts)

        try:
            import litellm

            response = litellm.embedding(
                model=self.config.embedding_model,
                input=texts,
            )
            return [item["embedding"] for item in response.data]
        except Exception as e:
            logger.warning(f"Failed to compute batch embeddings via LiteLLM: {e}")
            return [None] * len(texts)

    def _compute_embedding_sentence_transformers(
        self, text: str
    ) -> Optional[List[float]]:
        """Compute embedding using sentence-transformers (local)."""
        if not has_sentence_transformers():
            logger.warning("sentence-transformers not available for embeddings")
            return None

        try:
            model = self._get_sentence_transformer_model()
            embedding = model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            logger.warning(
                f"Failed to compute embedding via sentence-transformers: {e}"
            )
            return None

    def _compute_embeddings_batch_sentence_transformers(
        self, texts: List[str]
    ) -> List[Optional[List[float]]]:
        """Batch compute embeddings using sentence-transformers."""
        if not has_sentence_transformers():
            logger.warning("sentence-transformers not available for embeddings")
            return [None] * len(texts)

        try:
            model = self._get_sentence_transformer_model()
            embeddings = model.encode(texts, convert_to_numpy=True)
            return [emb.tolist() for emb in embeddings]
        except Exception as e:
            logger.warning(
                f"Failed to compute batch embeddings via sentence-transformers: {e}"
            )
            return [None] * len(texts)

    def _get_sentence_transformer_model(self):
        """Lazy load sentence-transformers model."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.config.local_model_name)
        return self._model

    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two embedding vectors.

        Args:
            a: First embedding vector
            b: Second embedding vector

        Returns:
            Cosine similarity score between 0 and 1
        """
        if not has_numpy():
            # Fallback to pure Python
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

    def ensure_embeddings(self, playbook: "Playbook") -> int:
        """Ensure all active bullets have embeddings computed.

        Args:
            playbook: Playbook to process

        Returns:
            Number of embeddings computed
        """
        bullets_needing_embeddings = [
            b for b in playbook.bullets() if b.embedding is None
        ]

        if not bullets_needing_embeddings:
            return 0

        texts = [b.content for b in bullets_needing_embeddings]
        embeddings = self.compute_embeddings_batch(texts)

        count = 0
        for bullet, embedding in zip(bullets_needing_embeddings, embeddings):
            if embedding is not None:
                bullet.embedding = embedding
                count += 1

        logger.info(f"Computed {count} embeddings for bullets")
        return count

    def detect_similar_pairs(
        self,
        playbook: "Playbook",
        threshold: Optional[float] = None,
    ) -> List[Tuple["Bullet", "Bullet", float]]:
        """Find all pairs of bullets with similarity >= threshold.

        Args:
            playbook: Playbook to search
            threshold: Similarity threshold (default: config.similarity_threshold)

        Returns:
            List of (bullet_a, bullet_b, similarity_score) tuples,
            sorted by similarity score descending
        """
        threshold = threshold or self.config.similarity_threshold
        similar_pairs: List[Tuple["Bullet", "Bullet", float]] = []

        # Get active bullets only
        bullets = playbook.bullets(include_invalid=False)

        # Group by section if configured
        if self.config.within_section_only:
            sections: dict[str, list] = {}
            for bullet in bullets:
                sections.setdefault(bullet.section, []).append(bullet)

            for section_bullets in sections.values():
                pairs = self._find_similar_in_list(section_bullets, playbook, threshold)
                similar_pairs.extend(pairs)
        else:
            similar_pairs = self._find_similar_in_list(bullets, playbook, threshold)

        # Sort by similarity descending
        similar_pairs.sort(key=lambda x: x[2], reverse=True)
        return similar_pairs

    def _find_similar_in_list(
        self,
        bullets: List["Bullet"],
        playbook: "Playbook",
        threshold: float,
    ) -> List[Tuple["Bullet", "Bullet", float]]:
        """Find similar pairs within a list of bullets."""
        pairs: List[Tuple["Bullet", "Bullet", float]] = []

        for i, bullet_a in enumerate(bullets):
            if bullet_a.embedding is None:
                continue

            for bullet_b in bullets[i + 1 :]:
                if bullet_b.embedding is None:
                    continue

                # Skip pairs with existing KEEP decisions
                if playbook.has_keep_decision(bullet_a.id, bullet_b.id):
                    continue

                similarity = self.cosine_similarity(
                    bullet_a.embedding, bullet_b.embedding
                )

                if similarity >= threshold:
                    pairs.append((bullet_a, bullet_b, similarity))

        return pairs
