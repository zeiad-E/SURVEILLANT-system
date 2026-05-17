"""
modules/reconciliation/worker.py
---------------------------------
Background Reconciliation Worker (Part B Improvement 3).

Runs every RECONCILIATION_INTERVAL_SEC seconds in a daemon thread.
Tasks:
  1. Detect merge candidates (persons whose galleries are very similar)
  2. Auto-merge if similarity >= AUTO_MERGE_THRESHOLD
  3. Propose manual review if below that threshold
  4. Promote 'unverified' persons to 'confirmed' when gallery >= 2
  5. Detect suspicious same-camera duplicates
"""

import time
import threading
import datetime
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations
from typing import Dict, Any

from modules.embedding.gallery import GalleryManager

from config.settings import (
    RECONCILIATION_INTERVAL_SEC,
    MERGE_CANDIDATE_THRESHOLD,
    AUTO_MERGE_THRESHOLD,
    GHOST_TTL_SEC,
    MIN_GALLERY_FOR_RECONCILIATION,
    MIN_VIEW_COVERAGE_FOR_MATCHING,
)


class ReconciliationWorker:
    """
    Periodically scans the database for inconsistencies and fixes them.
    Does NOT touch the real-time loop directly — only writes to DB.
    """

    def __init__(self) -> None:
        self._stop_event  = threading.Event()
        self._gallery_mgr = GalleryManager()

    def run_forever(self, db, track_registry: dict, color_registry=None, registry_lock=None) -> None:
        """
        Main loop. Runs run_cycle() every RECONCILIATION_INTERVAL_SEC seconds.
        Designed to run as a daemon thread.
        """
        while not self._stop_event.is_set():
            time.sleep(RECONCILIATION_INTERVAL_SEC)
            try:
                summary = self.run_cycle(db, track_registry, color_registry, registry_lock)
                self._print_summary(summary)
            except Exception as exc:
                import traceback
                print(f"[RECONCILE] ERROR in cycle: {exc}\n{traceback.format_exc()}")

    def stop(self) -> None:
        self._stop_event.set()

    # ----------------------------------------------------------------
    # Main cycle
    # ----------------------------------------------------------------

    def run_cycle(
        self,
        db,
        track_registry: dict,
        color_registry=None,
        registry_lock=None,
    ) -> Dict[str, Any]:
        """
        Runs one full reconciliation cycle. Returns a summary dict.
        """
        t_start = time.time()

        merge_proposals  = 0
        auto_merges      = 0
        ghosts_marked    = 0
        promotions       = 0
        duplicates_found = 0

        # ── Task 1 & 4: Find merge candidates (all pairs + same-cam duplicates) ──
        all_galleries = db.get_all_galleries_typed()
        person_ids    = list(all_galleries.keys())

        for pid_a, pid_b in combinations(person_ids, 2):
            gallery_a = all_galleries[pid_a]
            gallery_b = all_galleries[pid_b]

            # Skip persons with too few embeddings — a 1-2 embedding prototype
            # is too noisy to be a reliable merge target and generates many
            # false proposals.
            if len(gallery_a) < MIN_GALLERY_FOR_RECONCILIATION:
                continue
            if len(gallery_b) < MIN_GALLERY_FOR_RECONCILIATION:
                continue

            # Skip persons without sufficient view diversity — single-angle
            # galleries produce false matches when viewed from the same angle
            # by chance (same hallway, same camera height).
            if self._gallery_mgr.get_view_coverage(gallery_a) < MIN_VIEW_COVERAGE_FOR_MATCHING:
                continue
            if self._gallery_mgr.get_view_coverage(gallery_b) < MIN_VIEW_COVERAGE_FOR_MATCHING:
                continue

            # Mean-pool similarity: average across ALL compatible embedding pairs.
            # Max-pool (old behaviour) took the single best score, so one
            # accidentally-similar pair out of 25 was enough for a false proposal.
            # Mean-pool requires CONSISTENT similarity — a legitimate merge will
            # score high across most pairs; a false positive will not.
            score = self._mean_pool_similarity(gallery_a, gallery_b)
            if score < MERGE_CANDIDATE_THRESHOLD:
                continue

            # Task 4 — check same-camera duplicate
            cams_a = set(db.get_cameras_for_person(pid_a))
            cams_b = set(db.get_cameras_for_person(pid_b))
            shared_cams = cams_a & cams_b
            if shared_cams:
                duplicates_found += 1
                print(
                    f"[RECONCILE] SUSPICIOUS DUPLICATE on cam{list(shared_cams)[0]}: "
                    f"person {pid_a[:8]} <-> {pid_b[:8]} sim={score:.3f}"
                )

            if score >= AUTO_MERGE_THRESHOLD:
                # Auto-merge: keep whichever was created first
                pdata_a = db.get_person(pid_a)
                pdata_b = db.get_person(pid_b)
                if not pdata_a or not pdata_b:
                    continue

                keep_id   = pid_a if (pdata_a.get("created_at", "") <= pdata_b.get("created_at", "")) else pid_b
                remove_id = pid_b if keep_id == pid_a else pid_a

                moved = db.merge_persons(keep_id, remove_id)

                # Update color_registry aliases (snapshot then apply)
                if color_registry is not None:
                    for key, pid in list(color_registry._aliases.items()):
                        if pid == remove_id:
                            color_registry._aliases[key] = keep_id

                # Update track_registry under lock to prevent detection-thread races
                if registry_lock:
                    with registry_lock:
                        for k, pid in list(track_registry.items()):
                            if pid == remove_id:
                                track_registry[k] = keep_id
                else:
                    for k, pid in list(track_registry.items()):
                        if pid == remove_id:
                            track_registry[k] = keep_id

                auto_merges += 1
                print(
                    f"[MERGE] person {remove_id[:8]} merged INTO {keep_id[:8]} "
                    f"(gallery: {moved} embeddings combined)"
                )
            else:
                db.propose_merge(pid_a, pid_b, score)
                merge_proposals += 1
                print(
                    f"[RECONCILE] MERGE CANDIDATE: person {pid_a[:8]} <-> {pid_b[:8]} "
                    f"sim={score:.3f}"
                )

        # ── Task 2: Mark ghost tracks (persons not seen recently) ──
        now = datetime.datetime.now()
        for person_data in db.get_all_persons():
            pid        = person_data.get("person_id")
            status     = person_data.get("status", "unverified")
            last_seen  = person_data.get("last_seen_time")

            if status == "ghost" or not last_seen:
                continue

            # Check if person is still in track_registry (hold lock for thread-safe read)
            if registry_lock:
                with registry_lock:
                    still_active = any(v == pid for v in track_registry.values())
            else:
                still_active = any(v == pid for v in track_registry.values())
            if still_active:
                continue

            try:
                last_dt  = datetime.datetime.fromisoformat(last_seen)
                elapsed  = (now - last_dt).total_seconds()
            except (ValueError, TypeError):
                continue

            if elapsed > GHOST_TTL_SEC:
                db.update_person_status(pid, "ghost")
                ghosts_marked += 1
                print(f"[RECONCILE] person {pid[:8]} marked inactive (not seen for {int(elapsed)}s)")

        # ── Task 3: Promote unverified → confirmed ──
        for person_data in db.get_persons_by_status("unverified"):
            pid          = person_data.get("person_id")
            gallery_size = db.get_gallery_size(pid)
            if gallery_size >= 2:
                db.update_person_status(pid, "confirmed")
                promotions += 1
                print(
                    f"[RECONCILE] person {pid[:8]} promoted unverified -> confirmed "
                    f"(gallery={gallery_size})"
                )

        duration_ms = (time.time() - t_start) * 1000
        return {
            "merge_proposals":  merge_proposals,
            "auto_merges":      auto_merges,
            "ghosts_marked":    ghosts_marked,
            "promotions":       promotions,
            "duplicates_found": duplicates_found,
            "cycle_duration_ms": duration_ms,
        }

    # ----------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------

    def _mean_pool_similarity(
        self,
        gallery_a: list,
        gallery_b: list,
    ) -> float:
        """
        Mean-pool similarity between two galleries.

        score = mean of cosine_similarity(a_i, b_j) over all compatible pairs.

        Why mean instead of max:
            Max-pool asks "is there any pair above threshold?" — one lucky pair
            out of N×M is enough to trigger a false proposal.
            Mean-pool asks "are these galleries consistently similar?" — a real
            duplicate will score high on most pairs; a false positive will not.

        Returns 0.0 if there are no compatible pairs (mismatched backbone dims).
        """
        scores = []
        for entry_a in gallery_a:
            vec_a = entry_a["embedding"].reshape(1, -1)
            for entry_b in gallery_b:
                vec_b = entry_b["embedding"]
                if vec_b.shape[0] != entry_a["embedding"].shape[0]:
                    continue
                sim = float(cosine_similarity(vec_a, vec_b.reshape(1, -1))[0][0])
                scores.append(sim)
        return float(np.mean(scores)) if scores else 0.0

    def _print_summary(self, summary: Dict[str, Any]) -> None:
        print("=" * 44)
        print("[RECONCILE CYCLE]")
        print(f"  Merge proposals:    {summary['merge_proposals']}")
        print(f"  Auto-merges:        {summary['auto_merges']}  (sim >= {AUTO_MERGE_THRESHOLD})")
        print(f"  Ghosts marked:      {summary['ghosts_marked']}")
        print(f"  Status promotions:  {summary['promotions']}")
        print(f"  Cycle time:       {summary['cycle_duration_ms']:.0f}ms")
        print("=" * 44)
