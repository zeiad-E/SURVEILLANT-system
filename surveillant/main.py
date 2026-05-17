"""
main.py — SURVEILLANT entry point
----------------------------------
Usage:
    Phase 1 (detection + tracking + display):
        python main.py --phase 1 --videos data/videos/cam1.mp4 ...

Architecture (Phase 1) — CPU-optimised:
    - MAIN THREAD   : reads frames at native FPS and renders the grid
    - WORKER THREAD : processes ONE camera per iteration in round-robin order
                      so each camera gets a fresh detection every N × T_detect ms
                      (far faster than running all 5 cameras sequentially)
    This keeps the display perfectly smooth while maximising detection
    throughput on CPU hardware.
"""

import argparse
import sys
import threading
import time
import cv2
import numpy as np

from modules.camera.simulator import CameraSimulator
from modules.detection.detector import PersonDetector
from modules.tracking.tracker import PersonTracker
from display.visualizer import GridDisplay
from config.settings import (
    FPS_TARGET,
    YOLO_MODEL,
    DETECTION_CONF,
    DETECTION_IMGSZ,
    GRID_COLS,
    DISPLAY_WIDTH,
    DISPLAY_HEIGHT,
)


# ---------------------------------------------------------------------------
# Phase 1 — Camera Simulator + Detection + Tracking + Display
# ---------------------------------------------------------------------------

def run_phase1(video_paths: list) -> None:
    """
    Run the Phase 1 pipeline.

    Display thread : smooth video at native FPS, draws last-known bounding boxes.
    Worker thread  : round-robin detection — one camera per cycle — so YOLO
                     runs as fast as possible without blocking the display.

    Press 'q' or close the window to stop.
    """
    num_cams = len(video_paths)
    print(f"[SURVEILLANT] Phase 1 starting — {num_cams} camera(s) | CPU mode.")

    simulator = CameraSimulator(video_paths, FPS_TARGET)
    detector  = PersonDetector(YOLO_MODEL, DETECTION_CONF, DETECTION_IMGSZ)
    trackers  = {i: PersonTracker(i) for i in range(num_cams)}
    display   = GridDisplay(num_cams, GRID_COLS, DISPLAY_WIDTH, DISPLAY_HEIGHT)

    simulator.start()
    print("[SURVEILLANT] All cameras open. Press 'q' or close window to quit.\n")

    # ----------------------------------------------------------------
    # Shared state
    # ----------------------------------------------------------------
    latest_frames: dict = {}         # {cam_id: np.ndarray}  — written by main, read by worker
    latest_tracks: dict = {i: [] for i in range(num_cams)}  # written by worker, read by main
    frame_lock  = threading.Lock()
    track_lock  = threading.Lock()
    running     = threading.Event()
    running.set()

    # ----------------------------------------------------------------
    # Worker — round-robin: ONE camera per cycle
    # ----------------------------------------------------------------
    def detection_worker():
        """
        Cycles through cameras one at a time, running detection + tracking.
        This is orders of magnitude faster than processing all cameras per
        loop on CPU because YOLO only loads a single frame at a time.
        """
        cam_ids = list(range(num_cams))
        idx = 0

        while running.is_set():
            cam_id = cam_ids[idx % len(cam_ids)]
            idx += 1

            # Grab the latest frame for this camera
            with frame_lock:
                frame = latest_frames.get(cam_id)

            if frame is None:
                time.sleep(0.005)
                continue

            try:
                detections = detector.detect(frame)
                tracks     = trackers[cam_id].update(detections, frame)

                with track_lock:
                    latest_tracks[cam_id] = tracks

                for t in tracks:
                    print(f"[CAM {cam_id}] Track {t['track_id']} @ {t['bbox']}")

            except Exception as exc:
                print(f"[ERROR] Camera {cam_id} detection error: {exc}")

    worker = threading.Thread(target=detection_worker, daemon=True)
    worker.start()

    # ----------------------------------------------------------------
    # Main loop — read + display at native video FPS
    # ----------------------------------------------------------------
    try:
        while not display.should_quit():
            frames = simulator.read_frames()
            if not frames:
                print("[SURVEILLANT] All camera feeds lost — exiting.")
                break

            # Publish fresh frames for the worker
            with frame_lock:
                latest_frames.update(frames)

            # Render with the latest known tracks (no blocking on worker)
            with track_lock:
                current_tracks = {k: list(v) for k, v in latest_tracks.items()}

            for cam_id, frame in frames.items():
                display.update(cam_id, frame, current_tracks.get(cam_id, []))

            display.render()

    except KeyboardInterrupt:
        print("\n[SURVEILLANT] Interrupted by user.")
    finally:
        running.clear()
        worker.join(timeout=2)
        simulator.release()
        cv2.destroyAllWindows()
        print("[SURVEILLANT] Phase 1 done.")


# ---------------------------------------------------------------------------
# Phase 2 — Database + Embedding + Cross-Camera Matching
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Phase 2 — Permanent Track Registry + Cross-Camera Matching + Reconciliation
# ---------------------------------------------------------------------------

def run_phase2(video_paths: list) -> None:
    """
    Phase 2 pipeline — Production-grade multi-camera person re-identification.

    Architecture:
      DETECTION THREAD  : round-robin (1 YOLO model, all cameras) — as fast as possible.
                          Saves crops for unidentified tracks and queues them for embedding.
      EMBEDDING THREAD  : async — identifies queued crops, writes to DB, updates registry.
      MAIN THREAD       : reads frames at native FPS, renders grid.
                          Filters out STALE tracks (no detection update > STALE_TRACK_TIMEOUT).

    Core rules:
      RULE 1 (intra-camera): Once (cam_id, track_id) is bound to a person_id,
                              that binding is PERMANENT. No re-checking.
      RULE 2 (inter-camera): When a brand-new track appears, embedding search is
                              done ONCE. Then Rule 1 applies.
    """
    import os
    import json
    import queue
    import datetime
    from display.visualizer import ColorRegistry
    from modules.storage.database import Database
    from modules.embedding.embedder import PersonEmbedder
    from modules.search.searcher import PersonSearcher
    from modules.embedding.gallery import GalleryManager
    from modules.reconciliation.worker import ReconciliationWorker
    from modules.preprocessing.quality_gate import CropQualityGate
    from modules.preprocessing.masking import (
        apply_mask_to_crop,
        associate_masks_to_tracks,
        _iou as _bbox_iou,
    )
    from config.settings import (
        NUM_FRAMES_FOR_EMBEDDING, SNAPSHOTS_DIR,
        BODY_MATCH_THRESHOLD, MAX_GALLERY_SIZE,
        TRACK_REGISTRY_PATH, RECONCILIATION_INTERVAL_SEC,
        STALE_TRACK_TIMEOUT, MIN_FRAMES_BETWEEN_SAMPLES,
        BYTETRACK_TRACK_THRESH,
    )

    num_cams = len(video_paths)
    print(f"[SURVEILLANT] Phase 2 starting — {num_cams} camera(s) | CPU mode.")

    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    TRACK_REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ── Shared objects ──────────────────────────────────────────────────────
    color_registry = ColorRegistry()
    db             = Database()
    embedder       = PersonEmbedder()
    searcher       = PersonSearcher(db, embedder)
    gallery_mgr    = GalleryManager()
    quality_gate   = CropQualityGate()    # Part 2 — preflight before queueing for identification

    # ── Track registry (RULE 1 store) ──────────────────────────────────────
    # (cam_id, track_id) → person_uuid   |   Permanent once written.
    # IMPORTANT: Do NOT load from the previous session file.
    # DeepSORT resets its track-ID counter every run, so old bindings would
    # silently attach new-session tracks to wrong person_uuids (Bug 3 fix).
    track_registry: dict = {}
    try:
        TRACK_REGISTRY_PATH.write_text(json.dumps({}))   # clear stale file
    except Exception:
        pass

    registry_lock = threading.Lock()

    def save_registry():
        serialized = {f"cam{k[0]}_track{k[1]}": v for k, v in track_registry.items()}
        TRACK_REGISTRY_PATH.write_text(json.dumps(serialized, indent=2))

    # ── Per-track in-memory state (used by display — no DB reads needed) ───
    # (cam_id, track_id) → {state, person_id, gallery_size, status}
    track_state_cache: dict = {}
    state_lock  = threading.Lock()
    flash_times: dict = {}     # (cam_id, track_id) → float timestamp of binding

    # ── Async embedding queue ───────────────────────────────────────────────
    # Items: ('identify', cam_id, track_id, [crop, ...])
    #        ('gallery',  cam_id, track_id, person_uuid, crop, bbox, prev_bbox)
    embed_queue  = queue.Queue(maxsize=40)
    pending_ids  = set()   # (cam_id, track_id) keys queued for identification
    pending_lock = threading.Lock()

    # ── Crop buffers (detection thread only) ───────────────────────────────
    crop_buffer:   dict = {}  # (cam_id, track_id) → list[crop]
    frame_counter: dict = {}  # (cam_id, track_id) → int
    prev_bboxes:   dict = {}  # (cam_id, track_id) → [x1,y1,x2,y2]  (Part 6 pose)

    # ── Per-camera last-detection timestamp (for stale track filter) ────────
    last_det_time: dict = {i: 0.0 for i in range(num_cams)}
    det_time_lock = threading.Lock()

    # ── Currently active track IDs per camera (for smart same-camera guard) ──
    # Updated each detection cycle so the guard only blocks LIVE conflicts.
    active_tracks_per_cam: dict = {i: set() for i in range(num_cams)}
    active_tracks_lock = threading.Lock()

    # ── Last DB last_seen write time (rate-limit DB writes) ─────────────────
    last_db_write: dict = {}  # person_uuid → float

    # ── Simulator / display (single shared detector) ────────────────────────
    simulator = CameraSimulator(video_paths, FPS_TARGET)
    # ONE shared detector → no CPU contention between cameras
    shared_detector = PersonDetector(YOLO_MODEL, DETECTION_CONF, DETECTION_IMGSZ)
    trackers = {i: PersonTracker(i) for i in range(num_cams)}
    display  = GridDisplay(num_cams, GRID_COLS, DISPLAY_WIDTH, DISPLAY_HEIGHT,
                           color_registry=color_registry)

    simulator.start()
    print("[SURVEILLANT] All cameras open. Press 'q' or close window to quit.\n")

    latest_frames: dict = {}
    latest_tracks: dict = {i: [] for i in range(num_cams)}
    frame_lock = threading.Lock()
    track_lock = threading.Lock()
    running    = threading.Event()
    running.set()

    # ── Start reconciliation worker ──────────────────────────────────────────
    recon_worker = ReconciliationWorker()
    recon_thread = threading.Thread(
        target=recon_worker.run_forever,
        args=(db, track_registry, color_registry, registry_lock),
        daemon=True,
    )
    recon_thread.start()
    print(f"[SURVEILLANT] Reconciliation worker started (interval={RECONCILIATION_INTERVAL_SEC}s).")

    # ─────────────────────────────────────────────────────────────────────────
    # DETECTION THREAD — round-robin, fast loop, NO DB reads, NO embedding
    # ─────────────────────────────────────────────────────────────────────────
    def detection_worker():
        cam_ids = list(range(num_cams))
        idx = 0

        while running.is_set():
            cam_id = cam_ids[idx % len(cam_ids)]
            idx += 1

            with frame_lock:
                frame = latest_frames.get(cam_id)

            if frame is None:
                continue   # no sleep — keep cycling

            try:
                detections = shared_detector.detect(frame)
                tracks     = trackers[cam_id].update(detections, frame)
                now_ts     = time.time()

                # Part 4 — recover per-track segmentation masks via IoU.
                # DeepSORT reorders detections, so index alignment is unsafe.
                # Tracks coasting through occlusion get no mask this frame
                # and fall back to the raw crop (handled inside apply_mask_to_crop).
                track_masks = associate_masks_to_tracks(tracks, detections)

                # Track which IDs are alive right now (used by same-camera guard)
                current_ids = {t["track_id"] for t in tracks}
                with active_tracks_lock:
                    active_tracks_per_cam[cam_id] = current_ids

                # Purge state for tracks that the tracker dropped this cycle
                active_keys = {(cam_id, tid) for tid in current_ids}
                for k in list(frame_counter):
                    if k[0] == cam_id and k not in active_keys:
                        frame_counter.pop(k, None)
                        crop_buffer.pop(k, None)
                        prev_bboxes.pop(k, None)   # Part 6

                for t in tracks:
                    track_id = t["track_id"]
                    key      = (cam_id, track_id)
                    frame_counter[key] = frame_counter.get(key, 0) + 1

                    x1, y1, x2, y2 = t["bbox"]
                    h, w = frame.shape[:2]
                    raw_crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

                    # We assess quality on the RAW crop so the darkness check
                    # measures the captured image, not the post-mask gray field.
                    # Masking happens later, after the gate decides to keep it.
                    track_mask = track_masks.get(track_id)

                    with registry_lock:
                        person_uuid = track_registry.get(key)

                    if person_uuid is not None:
                        # ── BOUND: annotate from cache, queue gallery update ──
                        with state_lock:
                            info = track_state_cache.get(key) or {}

                        flash_t = flash_times.get(key, 0.0)
                        if now_ts - flash_t < 1.0:
                            t["state"] = "flash_green"
                        else:
                            t["state"] = info.get("state", "new")

                        t["person_id"]   = person_uuid
                        t["gallery_size"]= info.get("gallery_size", 1)
                        t["status"]      = info.get("status", "unverified")

                        # Queue gallery update every N frames (non-blocking).
                        # Pre-gate on raw crop (darkness check must see original).
                        # Pass bbox/prev_bbox so the gallery can do pose-aware
                        # canonical-view force-accept (Part 6).
                        if (frame_counter[key] % MIN_FRAMES_BETWEEN_SAMPLES == 0
                                and raw_crop.size > 0):
                            quality = quality_gate.assess(raw_crop)
                            if quality.passes:
                                masked_crop = apply_mask_to_crop(raw_crop, track_mask)
                                cur_bbox  = t["bbox"]
                                prev_bbox = prev_bboxes.get(key)
                                try:
                                    embed_queue.put_nowait((
                                        "gallery", cam_id, track_id, person_uuid,
                                        masked_crop.copy(), cur_bbox, prev_bbox,
                                    ))
                                except queue.Full:
                                    pass

                    else:
                        # ── COLLECTING or PENDING IDENTIFICATION ──
                        with pending_lock:
                            is_pending = key in pending_ids

                        buf_len = len(crop_buffer.get(key) or [])
                        t["state"]        = "collecting"
                        t["buffer_len"]   = buf_len
                        t["buffer_total"] = NUM_FRAMES_FOR_EMBEDDING

                        if not is_pending:
                            buf = crop_buffer.setdefault(key, [])
                            if raw_crop.size > 0 and len(buf) < NUM_FRAMES_FOR_EMBEDDING:
                                # Part 7 — only buffer crops from HIGH-confidence
                                # detections. ByteTrack passes low-conf detections
                                # (>= 0.10) for second-stage association, but those
                                # noisy crops must NOT seed an identity.
                                # Bbox equality fails on Kalman-corrected tracks, so
                                # match by IoU against the original detections.
                                best_conf = 0.0
                                for d in detections:
                                    if _bbox_iou(d["bbox"], t["bbox"]) >= 0.7:
                                        if d["confidence"] > best_conf:
                                            best_conf = d["confidence"]
                                # If no detection matches the track (pure Kalman
                                # prediction this frame), skip — re-emerging tracks
                                # will be high-conf on their next detection cycle.
                                if best_conf >= BYTETRACK_TRACK_THRESH:
                                    masked_crop = apply_mask_to_crop(raw_crop, track_mask)
                                    buf.append(masked_crop.copy())

                            if len(buf) >= NUM_FRAMES_FOR_EMBEDDING:
                                # Capture pose at identification time so the
                                # initial embedding gets tagged with a canonical
                                # view (otherwise it gets "initial" which blocks
                                # reconciliation coverage scoring).
                                id_bbox      = t["bbox"]
                                id_prev_bbox = prev_bboxes.get(key)
                                try:
                                    embed_queue.put_nowait((
                                        "identify", cam_id, track_id, list(buf),
                                        id_bbox, id_prev_bbox,
                                    ))
                                    with pending_lock:
                                        pending_ids.add(key)
                                    crop_buffer[key] = []   # reset
                                    print(
                                        f"[BUFFER]  cam{cam_id}_track{track_id} "
                                        f"full — queued for embedding"
                                    )
                                except queue.Full:
                                    pass   # will retry next cycle

                        # Part 6 — store bbox for next-cycle pose estimation
                        prev_bboxes[key] = t["bbox"]

                with track_lock:
                    latest_tracks[cam_id] = tracks

                with det_time_lock:
                    last_det_time[cam_id] = now_ts

            except Exception as exc:
                import traceback
                print(f"[ERROR] cam{cam_id} detection: {exc}\n{traceback.format_exc()}")

    # ─────────────────────────────────────────────────────────────────────────
    # EMBEDDING THREAD — slow ops: inference, DB reads/writes, gallery updates
    # ─────────────────────────────────────────────────────────────────────────
    def embedding_worker():
        while running.is_set():
            try:
                item = embed_queue.get(timeout=0.3)
            except queue.Empty:
                continue

            task = item[0]

            if task == "gallery":
                # Unpack: task, cam_id, track_id, person_uuid, crop, bbox, prev_bbox
                _, cam_id, track_id, person_uuid, crop, g_bbox, g_prev_bbox = item
                key = (cam_id, track_id)
                fc  = frame_counter.get(key, 0)
                gallery_mgr.maybe_update_gallery(
                    person_id   = person_uuid,
                    crop        = crop,
                    embedder    = embedder,
                    db          = db,
                    frame_count = fc,
                    cam_id      = cam_id,
                    bbox        = g_bbox,        # Part 6 — for pose-aware view classification
                    prev_bbox   = g_prev_bbox,
                )
                # Update gallery_size in state cache
                new_size = db.get_gallery_size(person_uuid)
                new_status = (db.get_person(person_uuid) or {}).get("status", "unverified")
                with state_lock:
                    if key in track_state_cache:
                        track_state_cache[key]["gallery_size"] = new_size
                        track_state_cache[key]["status"]       = new_status

                # Rate-limited last_seen DB write (max once per 5 sec per person)
                now_ts = time.time()
                if now_ts - last_db_write.get(person_uuid, 0.0) > 5.0:
                    last_db_write[person_uuid] = now_ts
                    now_str = datetime.datetime.now().isoformat()
                    db.update_last_seen(person_uuid, cam_id, now_str)
                    print(
                        f"[UPDATE]  person {person_uuid[:8]} | "
                        f"last_seen -> cam{cam_id} @ {now_str}"
                    )

            elif task == "identify":
                # Unpack: task, cam_id, track_id, buf, id_bbox, id_prev_bbox
                _, cam_id, track_id, buf, id_bbox, id_prev_bbox = item
                key = (cam_id, track_id)

                try:
                    # Double-check not already identified
                    with registry_lock:
                        if key in track_registry:
                            continue

                    now_str = datetime.datetime.now().isoformat()

                    embs      = [embedder.extract_body_embedding(cr) for cr in buf]
                    final_emb = embedder.aggregate_embeddings(embs)

                    matches = searcher.search_by_embedding(
                        final_emb, query_embedding_type="body", top_k=1
                    )

                    if matches and matches[0]["similarity_score"] >= BODY_MATCH_THRESHOLD:
                        candidate_uuid = matches[0]["person_id"]

                        # ── Same-camera guard ──────────────────────────────
                        # A match is a false positive if another LIVE track on
                        # the same camera is already bound to the same person.
                        # We only block when the conflicting track is currently
                        # active — dead tracks in the registry don't count,
                        # otherwise a returning person (whose old track died)
                        # would always be forced to create a new person_id.
                        with active_tracks_lock:
                            live_ids = set(active_tracks_per_cam.get(cam_id, set()))
                        with registry_lock:
                            same_cam_conflict = any(
                                pid == candidate_uuid
                                and k[0] == cam_id
                                and k[1] != track_id
                                and k[1] in live_ids   # only block LIVE conflicts
                                for k, pid in track_registry.items()
                            )

                        if same_cam_conflict:
                            print(
                                f"[GUARD]   cam{cam_id}_track{track_id} -> same-camera conflict "
                                f"with {candidate_uuid[:8]} — creating NEW person"
                            )
                            matches = []   # fall through to new-person branch
                        else:
                            person_uuid = candidate_uuid
                            status_out  = "returning"
                            db.update_last_seen(person_uuid, cam_id, now_str)
                            known_cams = db.get_cameras_for_person(person_uuid)
                            db.upsert_camera_history(person_uuid, cam_id, track_id, now_str)
                            if any(c != cam_id for c in known_cams):
                                db.update_person_status(person_uuid, "multi_view")
                            print(
                                f"[MATCH]   cam{cam_id}_track{track_id} -> EXISTING person "
                                f"{person_uuid[:8]} body_sim={matches[0]['similarity_score']:.2f}"
                            )

                    if not matches:
                        # Brand-new person — tag the initial embedding with a
                        # CANONICAL view so reconciliation's view-coverage gate
                        # can count it (otherwise "initial" leaves coverage=0).
                        from modules.embedding.gallery import estimate_view
                        canonical = estimate_view(id_bbox, id_prev_bbox)
                        person_uuid = db.insert_person({
                            "cam_id"         : cam_id,
                            "embedding"      : embedder.serialize(final_emb),
                            "embedding_type" : "body",
                            "angle_tag"      : canonical,
                            "first_seen_cam" : cam_id,
                            "first_seen_time": now_str,
                            "last_seen_cam"  : cam_id,
                            "last_seen_time" : now_str,
                            "snapshot_paths" : [],
                            "created_at"     : now_str,
                        })
                        status_out = "new"
                        db.upsert_camera_history(person_uuid, cam_id, track_id, now_str)
                        print(
                            f"[NEW]     cam{cam_id}_track{track_id} -> NEW person "
                            f"{person_uuid[:8]} (no match above threshold)"
                        )

                    # ── Write to registry (RULE 1) ──
                    with registry_lock:
                        if key not in track_registry:
                            track_registry[key] = person_uuid

                    color_registry.register_alias(cam_id, track_id, person_uuid)
                    save_registry()

                    # Init display state cache
                    person_data = db.get_person(person_uuid)
                    with state_lock:
                        track_state_cache[key] = {
                            "state"      : status_out,
                            "person_id"  : person_uuid,
                            "gallery_size": db.get_gallery_size(person_uuid),
                            "status"     : (person_data or {}).get("status", "unverified"),
                        }
                    flash_times[key] = time.time()

                    # Save snapshot crops
                    person_folder  = SNAPSHOTS_DIR / person_uuid
                    person_folder.mkdir(exist_ok=True)
                    existing_files = len(list(person_folder.glob("*.jpg")))
                    saved_paths    = []
                    for i, cr in enumerate(buf):
                        path = person_folder / f"crop_{existing_files + i}.jpg"
                        cv2.imwrite(str(path), cr)
                        saved_paths.append(str(path))

                    p_data = db.get_person(person_uuid)
                    if p_data:
                        all_paths = (p_data.get("snapshot_paths") or []) + saved_paths
                        with db._get_conn() as conn:
                            conn.execute(
                                "UPDATE persons SET snapshot_paths=? WHERE person_id=?",
                                (json.dumps(all_paths), person_uuid),
                            )

                    # Reinforce DeepSORT appearance model (RULE 4)
                    gallery_vecs = db.get_gallery(person_uuid)
                    if gallery_vecs:
                        trackers[cam_id].reinforce_track(track_id, person_uuid, gallery_vecs)

                except Exception as exc:
                    import traceback
                    print(
                        f"[EMBED ERROR] cam{cam_id}_track{track_id}: "
                        f"{exc}\n{traceback.format_exc()}"
                    )
                finally:
                    with pending_lock:
                        pending_ids.discard(key)

    # ── Start threads ──────────────────────────────────────────────────────
    det_thread  = threading.Thread(target=detection_worker, daemon=True)
    emb_thread  = threading.Thread(target=embedding_worker, daemon=True)
    det_thread.start()
    emb_thread.start()
    print("[SURVEILLANT] Detection (round-robin) + embedding (async) workers started.")

    # ─────────────────────────────────────────────────────────────────────────
    # MAIN LOOP — read frames at native FPS, render display
    # ─────────────────────────────────────────────────────────────────────────
    try:
        while not display.should_quit():
            frames = simulator.read_frames()
            if not frames:
                print("[SURVEILLANT] All camera feeds lost — exiting.")
                break

            with frame_lock:
                latest_frames.update(frames)

            now = time.time()
            with track_lock:
                current_tracks = {k: list(v) for k, v in latest_tracks.items()}
            with det_time_lock:
                current_det_times = dict(last_det_time)

            for cam_id, frame in frames.items():
                tracks = current_tracks.get(cam_id, [])

                # ── STALE TRACK FILTER (ghost box fix) ──
                # If detection hasn't run for this camera in STALE_TRACK_TIMEOUT seconds,
                # the tracks in latest_tracks are outdated — clear them so frozen boxes
                # don't appear on screen.
                if now - current_det_times.get(cam_id, 0.0) > STALE_TRACK_TIMEOUT:
                    tracks = []

                display.update(cam_id, frame, tracks)

            display.render()

    except KeyboardInterrupt:
        print("\n[SURVEILLANT] Interrupted by user.")
    finally:
        running.clear()
        recon_worker.stop()
        det_thread.join(timeout=2)
        emb_thread.join(timeout=5)
        simulator.release()
        cv2.destroyAllWindows()
        save_registry()
        print("[SURVEILLANT] Phase 2 done.")



# ---------------------------------------------------------------------------
# Phase 3 — Photo Search CLI
# ---------------------------------------------------------------------------


def run_phase3(query_path: str) -> None:
    """
    Offline Photo Search.
    Extracts features from the query image, searches SQLite, and displays results.
    """
    import os
    import math
    from modules.storage.database import Database
    from modules.embedding.embedder import PersonEmbedder
    from modules.search.searcher import PersonSearcher

    if not os.path.exists(query_path):
        print(f"[ERROR] Query image not found: {query_path}")
        return

    print(f"[SURVEILLANT] Phase 3: Searching database for {query_path}...")
    db = Database()
    embedder = PersonEmbedder()
    searcher = PersonSearcher(db, embedder)

    matches = searcher.search_by_photo(query_path, top_k=5)
    if not matches:
        print("[SURVEILLANT] No matches found in the database.")
        return

    print("--- SEARCH RESULTS ---")
    
    # Load original image for display
    query_img_orig = cv2.imread(query_path)
    
    # Prepare display canvas. We will show the query image on the left,
    # and the top 5 matches stacked vertically or in a grid on the right.
    cv2.namedWindow("SURVEILLANT - Search Results", cv2.WINDOW_NORMAL)
    
    result_panels = []
    
    for i, match in enumerate(matches):
        score = match['similarity_score']
        pid = match['person_id']
        cam = match['last_seen_cam']
        time_seen = match['last_seen_time']
        
        print(f"#{i+1}: Score: {score:.2f} | Cam: {cam} | Time: {time_seen} | ID: {pid[:8]}")
        
        # Load best snapshot
        snapshots = match.get("snapshot_paths", [])
        if snapshots and os.path.exists(snapshots[0]):
            snapshot_img = cv2.imread(snapshots[0])
            snapshot_img = cv2.resize(snapshot_img, (200, 400))
        else:
            snapshot_img = np.zeros((400, 200, 3), dtype=np.uint8)

        # Add text overlay
        cv2.putText(snapshot_img, f"#{i+1} Match", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(snapshot_img, f"Score: {score:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(snapshot_img, f"Cam: {cam}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        result_panels.append(snapshot_img)

    # Pad if we have less than top_k results
    while len(result_panels) < 5:
        result_panels.append(np.zeros((400, 200, 3), dtype=np.uint8))

    # Compile the right side
    right_side = np.hstack(result_panels)
    
    # Resize query to match height of the right side (400px)
    query_display = cv2.resize(query_img_orig, (200, 400))
    cv2.putText(query_display, "QUERY", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    # Add a visual separator
    separator = np.full((400, 10, 3), 200, dtype=np.uint8)
    
    final_canvas = np.hstack([query_display, separator, right_side])
    
    cv2.imshow("SURVEILLANT - Search Results", final_canvas)
    print("\n[SURVEILLANT] Press any key in the image window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="surveillant",
        description="SURVEILLANT — AI-powered multi-camera surveillance system",
    )
    parser.add_argument(
        "--phase",
        type=int,
        required=True,
        choices=[1, 2, 3, 4, 5, 6],
        help="Pipeline phase to run (1–6).",
    )
    parser.add_argument(
        "--videos",
        nargs="+",
        metavar="VIDEO",
        help="Paths to video files (one per camera). Required for phase 1.",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Path to a query image. Required for phase 3.",
    )
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args   = parser.parse_args()

    if args.phase in [1, 2]:
        if not args.videos:
            parser.error("--videos is required for phase 1/2.")
        
        if args.phase == 1:
            run_phase1(args.videos)
        else:
            run_phase2(args.videos)

    elif args.phase == 3:
        if not args.query:
            parser.error("--query is required for phase 3.")
        run_phase3(args.query)

    else:
        print(f"[SURVEILLANT] Phase {args.phase} is not yet implemented.")
        sys.exit(1)
