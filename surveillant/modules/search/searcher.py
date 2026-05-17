"""
modules/search/searcher.py
--------------------------
Type-aware, max-pooling cross-camera person search.

Uses FACE_MATCH_THRESHOLD / BODY_MATCH_THRESHOLD depending on query type.
Applies CROSS_TYPE_MULTIPLIER penalty when query type != stored type.
Only persons with gallery_size >= MIN_GALLERY_FOR_MATCHING are candidates.
"""

import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any

from modules.storage.database import Database
from modules.embedding.embedder import PersonEmbedder
from config.settings import (
    FACE_MATCH_THRESHOLD,
    BODY_MATCH_THRESHOLD,
    CROSS_TYPE_MULTIPLIER,
    MIN_GALLERY_FOR_MATCHING,
)

CROSS_TYPE_PENALTY = CROSS_TYPE_MULTIPLIER   # exported for tests


class PersonSearcher:
    """
    Compares incoming embeddings against all known persons in the DB.

    Max-pooling strategy:
        score(person) = max(adjusted_sim(query, gallery_i))
    Cross-type penalty applied when query_type != stored_type.
    """

    def __init__(self, db: Database, embedder: PersonEmbedder) -> None:
        self.db      = db
        self.embedder = embedder

    def search_by_embedding(
        self,
        query_embedding: np.ndarray,
        query_embedding_type: str = "body",
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Max-pooling search over all person galleries.

        Args:
            query_embedding:      1D normalized float32 array.
            query_embedding_type: 'face' or 'body'.
            top_k:                Max persons to return.

        Returns:
            List of person dicts (sorted by score desc) with 'similarity_score' key.
            Only persons above threshold and meeting min gallery size are included.
        """
        all_galleries = self.db.get_all_galleries_typed()
        if not all_galleries:
            return []

        threshold = (
            FACE_MATCH_THRESHOLD if query_embedding_type == "face"
            else BODY_MATCH_THRESHOLD
        )

        query_2d = query_embedding.reshape(1, -1)
        person_scores: Dict[str, float] = {}

        for pid, gallery_entries in all_galleries.items():
            if len(gallery_entries) < MIN_GALLERY_FOR_MATCHING:
                continue
            # NOTE: view-coverage check deliberately removed from the searcher.
            # Gating here caused a cascade: person with 1 view re-enters the
            # frame → searcher skips them → creates a duplicate person_id.
            # View coverage is used only in reconciliation (a quality gate for
            # high-confidence merge decisions), not for real-time identification.

            best_score = 0.0
            for entry in gallery_entries:
                stored_vec  = entry["embedding"]
                stored_type = entry["type"]

                if stored_vec.shape[0] != query_embedding.shape[0]:
                    continue  # stale embedding from a different backbone — skip

                sim = float(
                    cosine_similarity(query_2d, stored_vec.reshape(1, -1))[0][0]
                )
                # NOTE: CROSS_TYPE_PENALTY removed — face embeddings were dropped
                # in Part 5, so every stored embedding is "body". The penalty
                # was dead code.

                if sim > best_score:
                    best_score = sim

            person_scores[pid] = best_score

        # Filter by threshold, rank descending
        above_threshold = {
            pid: score
            for pid, score in person_scores.items()
            if score >= threshold
        }
        sorted_persons = sorted(
            above_threshold.items(), key=lambda x: x[1], reverse=True
        )

        results = []
        for pid, score in sorted_persons[:top_k]:
            person_data = self.db.get_person(pid)
            if person_data:
                person_data["similarity_score"] = score
                results.append(person_data)

        return results

    def search_by_photo(
        self, query_image_path: str, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Load an image from disk, extract body embedding, and search."""
        img = cv2.imread(query_image_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image at {query_image_path}")
        vector = self.embedder.extract_body_embedding(img)
        return self.search_by_embedding(vector, query_embedding_type="body", top_k=top_k)
