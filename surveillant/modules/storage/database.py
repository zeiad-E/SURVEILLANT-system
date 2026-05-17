"""
modules/storage/database.py
----------------------------
SQLite wrapper for SURVEILLANT.

Schema (final version):
  persons          — one row per unique physical person
  person_embeddings — one row per gallery embedding (many per person)
  camera_history   — cross-camera sighting log
  merge_proposals  — reconciliation merge candidates
"""

import sqlite3
import json
import uuid
import datetime
import numpy as np
from contextlib import contextmanager
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

from config.settings import DB_PATH


class Database:
    """
    SQLite wrapper for the SURVEILLANT system.

    Supports both file-based and in-memory (':memory:') databases.
    In-memory mode is used by tests to avoid file-lock issues on Windows.
    """

    def __init__(self, db_path=DB_PATH) -> None:
        self._in_memory = str(db_path) == ":memory:"

        if self._in_memory:
            self.db_path = ":memory:"
            self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        else:
            self.db_path = str(Path(db_path))
            self._conn = None
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self._init_db()

    @contextmanager
    def _get_conn(self):
        """Context manager that yields the right connection."""
        if self._in_memory:
            yield self._conn
        else:
            with sqlite3.connect(self.db_path, timeout=15.0) as conn:
                conn.execute("PRAGMA journal_mode=WAL;")
                yield conn

    def _init_db(self) -> None:
        """Create / migrate the schema."""
        with self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS persons (
                    person_id           TEXT PRIMARY KEY,
                    first_seen_cam      INTEGER,
                    first_seen_time     TEXT,
                    last_seen_cam       INTEGER,
                    last_seen_time      TEXT,
                    status              TEXT DEFAULT 'unverified',
                    gallery_size        INTEGER DEFAULT 0,
                    known_angles        TEXT DEFAULT '[]',
                    last_gallery_update TEXT,
                    description         TEXT,
                    gender              TEXT,
                    age_range           TEXT,
                    snapshot_paths      TEXT DEFAULT '[]',
                    created_at          TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS person_embeddings (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id       TEXT NOT NULL,
                    embedding       BLOB NOT NULL,
                    embedding_type  TEXT NOT NULL,
                    angle_tag       TEXT DEFAULT 'unknown',
                    source_cam      INTEGER,
                    captured_at     TEXT NOT NULL,
                    FOREIGN KEY (person_id) REFERENCES persons(person_id)
                );

                CREATE TABLE IF NOT EXISTS camera_history (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id   TEXT NOT NULL,
                    cam_id      INTEGER NOT NULL,
                    track_id    INTEGER NOT NULL,
                    first_seen  TEXT,
                    last_seen   TEXT,
                    FOREIGN KEY (person_id) REFERENCES persons(person_id)
                );

                CREATE TABLE IF NOT EXISTS merge_proposals (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id_a     TEXT,
                    person_id_b     TEXT,
                    similarity      REAL,
                    proposed_at     TEXT,
                    status          TEXT DEFAULT 'pending',
                    resolved_at     TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_gallery_pid
                    ON person_embeddings(person_id);
                CREATE INDEX IF NOT EXISTS idx_cam_history_pid
                    ON camera_history(person_id);
            """)
            if self._in_memory:
                conn.commit()

        # Migrate legacy schemas (file-based DB that may pre-date this version)
        if not self._in_memory:
            self._migrate()

    def _migrate(self) -> None:
        """Add missing columns to existing databases without losing data."""
        migrations = [
            "ALTER TABLE persons ADD COLUMN status TEXT DEFAULT 'unverified'",
            "ALTER TABLE persons ADD COLUMN gallery_size INTEGER DEFAULT 0",
            "ALTER TABLE persons ADD COLUMN known_angles TEXT DEFAULT '[]'",
            "ALTER TABLE persons ADD COLUMN last_gallery_update TEXT",
            "ALTER TABLE person_embeddings ADD COLUMN source_cam INTEGER",
        ]
        with self._get_conn() as conn:
            for sql in migrations:
                try:
                    conn.execute(sql)
                except sqlite3.OperationalError:
                    pass  # column already exists

    # ------------------------------------------------------------------
    # Public API — Persons
    # ------------------------------------------------------------------

    def insert_person(self, record: Dict[str, Any]) -> str:
        """
        Insert a newly detected person.
        Accepts an optional 'person_id' key (for tests that need a known UUID).
        Seeds the gallery with the initial embedding if 'embedding' is provided.
        """
        person_id = record.get("person_id") or str(uuid.uuid4())
        snapshot_paths_str = json.dumps(record.get("snapshot_paths", []))
        now = record.get("created_at", datetime.datetime.now().isoformat())

        query = """
        INSERT OR IGNORE INTO persons (
            person_id, first_seen_cam, first_seen_time,
            last_seen_cam, last_seen_time,
            status, gallery_size, known_angles, last_gallery_update,
            description, gender, age_range,
            snapshot_paths, created_at
        ) VALUES (?, ?, ?, ?, ?, 'unverified', 0, '[]', NULL, ?, ?, ?, ?, ?)
        """
        values = (
            person_id,
            record.get("first_seen_cam", record.get("cam_id", 0)),
            record.get("first_seen_time", now),
            record.get("last_seen_cam",  record.get("cam_id", 0)),
            record.get("last_seen_time",  now),
            record.get("description"),
            record.get("gender"),
            record.get("age_range"),
            snapshot_paths_str,
            now,
        )
        with self._get_conn() as conn:
            conn.execute(query, values)
            if self._in_memory:
                conn.commit()

        # Seed gallery with initial embedding (if provided).
        # angle_tag defaults to "initial" but the caller can override with a
        # canonical view ("frontal", "side", "right_moving", "left_moving").
        # This matters: get_view_coverage() only counts canonical views, so
        # using "initial" leaves the person at 0.0 coverage and blocks
        # reconciliation forever.
        emb_bytes = record.get("embedding")
        if emb_bytes:
            self.add_embedding_to_gallery(
                person_id      = person_id,
                embedding_bytes= emb_bytes,
                embedding_type = record.get("embedding_type", "body"),
                angle_tag      = record.get("angle_tag", "initial"),
                source_cam     = record.get("cam_id", 0),
                captured_at    = now,
            )

        return person_id

    def get_all_persons(self) -> List[Dict[str, Any]]:
        """Return all person records as dictionaries."""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM persons")
            rows = cursor.fetchall()
        results = []
        for row in rows:
            d = dict(row)
            for field in ("snapshot_paths", "known_angles"):
                if d.get(field):
                    try:
                        d[field] = json.loads(d[field])
                    except (json.JSONDecodeError, TypeError):
                        d[field] = []
            results.append(d)
        return results

    def get_person(self, person_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single person by ID."""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM persons WHERE person_id = ?", (person_id,)
            )
            row = cursor.fetchone()
        if not row:
            return None
        d = dict(row)
        for field in ("snapshot_paths", "known_angles"):
            if d.get(field):
                try:
                    d[field] = json.loads(d[field])
                except (json.JSONDecodeError, TypeError):
                    d[field] = []
        return d

    def get_persons_by_status(self, status: str) -> List[Dict[str, Any]]:
        """Return all persons with the given status."""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM persons WHERE status = ?", (status,)
            )
            rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def update_last_seen(self, person_id: str, cam_id: int, timestamp: str) -> None:
        """Update last_seen when a person is re-identified."""
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE persons SET last_seen_cam=?, last_seen_time=? WHERE person_id=?",
                (cam_id, timestamp, person_id),
            )
            if self._in_memory:
                conn.commit()

    def update_person_status(self, person_id: str, status: str) -> None:
        """
        Update person status (unverified / confirmed / multi_view / flagged / ghost).
        Called automatically when gallery grows or cross-camera match found.
        """
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE persons SET status=? WHERE person_id=?",
                (status, person_id),
            )
            if self._in_memory:
                conn.commit()

    def update_description(
        self, person_id: str, description: str, gender: str, age_range: str
    ) -> None:
        """Update LLM-generated profile attributes."""
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE persons SET description=?, gender=?, age_range=? WHERE person_id=?",
                (description, gender, age_range, person_id),
            )
            if self._in_memory:
                conn.commit()

    # ------------------------------------------------------------------
    # Public API — Gallery
    # ------------------------------------------------------------------

    def add_embedding_to_gallery(
        self,
        person_id: str,
        embedding_bytes: bytes,
        embedding_type: str,
        angle_tag: str,
        captured_at: str,
        source_cam: int = 0,
    ) -> None:
        """
        Add a new view embedding to a person's gallery.
        Also updates denormalized gallery_size and known_angles on the person row,
        and automatically promotes the person status when gallery grows.
        """
        # Everything in ONE transaction to avoid nested-connection deadlocks on WAL.
        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO person_embeddings
                   (person_id, embedding, embedding_type, angle_tag, source_cam, captured_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (person_id, embedding_bytes, embedding_type, angle_tag, source_cam, captured_at),
            )

            cursor = conn.execute(
                "SELECT gallery_size, known_angles FROM persons WHERE person_id=?",
                (person_id,),
            )
            row = cursor.fetchone()
            if row:
                new_size = (row[0] or 0) + 1
                angles_list = json.loads(row[1] or "[]")
                if angle_tag not in angles_list:
                    angles_list.append(angle_tag)
                conn.execute(
                    """UPDATE persons
                       SET gallery_size=?, known_angles=?, last_gallery_update=?
                       WHERE person_id=?""",
                    (new_size, json.dumps(angles_list), captured_at, person_id),
                )
                # Inline status promotion — avoids opening a second connection inside this one
                if new_size == 2:
                    conn.execute(
                        "UPDATE persons SET status='confirmed' WHERE person_id=?",
                        (person_id,),
                    )

            if self._in_memory:
                conn.commit()

    def get_gallery(self, person_id: str) -> List[np.ndarray]:
        """Return all gallery embeddings as numpy arrays (vectors only)."""
        with self._get_conn() as conn:
            cursor = conn.execute(
                "SELECT embedding FROM person_embeddings WHERE person_id=? ORDER BY id ASC",
                (person_id,),
            )
            rows = cursor.fetchall()
        return [np.frombuffer(row[0], dtype=np.float32) for row in rows]

    def get_gallery_typed(self, person_id: str) -> List[Dict[str, Any]]:
        """Return gallery as list of {embedding, type, source_cam} dicts."""
        with self._get_conn() as conn:
            cursor = conn.execute(
                "SELECT embedding, embedding_type, source_cam FROM person_embeddings "
                "WHERE person_id=? ORDER BY id ASC",
                (person_id,),
            )
            rows = cursor.fetchall()
        return [
            {
                "embedding":  np.frombuffer(row[0], dtype=np.float32),
                "type":       row[1],
                "source_cam": row[2],
            }
            for row in rows
        ]

    def get_gallery_size(self, person_id: str) -> int:
        """Return number of embeddings in a person's gallery (uses denormalized column)."""
        with self._get_conn() as conn:
            cursor = conn.execute(
                "SELECT gallery_size FROM persons WHERE person_id=?", (person_id,)
            )
            row = cursor.fetchone()
            if row is not None:
                return row[0] or 0
            return 0

    def get_all_galleries_typed(self) -> Dict[str, List[Dict[str, Any]]]:
        """Return {person_id: [{embedding, type, source_cam}, ...]} for all persons."""
        with self._get_conn() as conn:
            cursor = conn.execute(
                "SELECT person_id, embedding, embedding_type, source_cam "
                "FROM person_embeddings ORDER BY person_id, id ASC"
            )
            rows = cursor.fetchall()
        galleries: Dict[str, List[Dict[str, Any]]] = {}
        for pid, emb_bytes, emb_type, src_cam in rows:
            arr = np.frombuffer(emb_bytes, dtype=np.float32)
            galleries.setdefault(pid, []).append(
                {"embedding": arr, "type": emb_type, "source_cam": src_cam}
            )
        return galleries

    def get_all_galleries(self) -> Dict[str, List[np.ndarray]]:
        """Return {person_id: [ndarray, ...]}. Legacy; prefer get_all_galleries_typed()."""
        with self._get_conn() as conn:
            cursor = conn.execute(
                "SELECT person_id, embedding FROM person_embeddings ORDER BY person_id, id ASC"
            )
            rows = cursor.fetchall()
        galleries: Dict[str, List[np.ndarray]] = {}
        for pid, emb_bytes in rows:
            galleries.setdefault(pid, []).append(np.frombuffer(emb_bytes, dtype=np.float32))
        return galleries

    # ------------------------------------------------------------------
    # Public API — Camera History
    # ------------------------------------------------------------------

    def upsert_camera_history(
        self, person_id: str, cam_id: int, track_id: int, timestamp: str
    ) -> None:
        """Record or update a camera sighting for a person."""
        with self._get_conn() as conn:
            cursor = conn.execute(
                "SELECT id FROM camera_history WHERE person_id=? AND cam_id=? AND track_id=?",
                (person_id, cam_id, track_id),
            )
            row = cursor.fetchone()
            if row:
                conn.execute(
                    "UPDATE camera_history SET last_seen=? WHERE id=?",
                    (timestamp, row[0]),
                )
            else:
                conn.execute(
                    "INSERT INTO camera_history (person_id, cam_id, track_id, first_seen, last_seen) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (person_id, cam_id, track_id, timestamp, timestamp),
                )
            if self._in_memory:
                conn.commit()

    def get_cameras_for_person(self, person_id: str) -> List[int]:
        """Return list of distinct cam_ids where this person has been seen."""
        with self._get_conn() as conn:
            cursor = conn.execute(
                "SELECT DISTINCT cam_id FROM camera_history WHERE person_id=?",
                (person_id,),
            )
            return [row[0] for row in cursor.fetchall()]

    # ------------------------------------------------------------------
    # Public API — Merge Proposals
    # ------------------------------------------------------------------

    def propose_merge(self, pid_a: str, pid_b: str, similarity: float) -> None:
        """
        Log a reconciliation merge proposal, deduplicating by person pair.

        If a pending proposal for (pid_a, pid_b) or (pid_b, pid_a) already
        exists, update its similarity score instead of inserting a duplicate.
        Without this, the same wrong pair would accumulate a new row every
        120 seconds, producing dozens of identical false proposals.
        """
        now = datetime.datetime.now().isoformat()
        with self._get_conn() as conn:
            existing = conn.execute(
                """SELECT id FROM merge_proposals
                   WHERE status = 'pending'
                     AND ((person_id_a = ? AND person_id_b = ?)
                          OR (person_id_a = ? AND person_id_b = ?))""",
                (pid_a, pid_b, pid_b, pid_a),
            ).fetchone()

            if existing:
                conn.execute(
                    "UPDATE merge_proposals SET similarity=?, proposed_at=? WHERE id=?",
                    (similarity, now, existing[0]),
                )
            else:
                conn.execute(
                    "INSERT INTO merge_proposals "
                    "(person_id_a, person_id_b, similarity, proposed_at) VALUES (?, ?, ?, ?)",
                    (pid_a, pid_b, similarity, now),
                )
            if self._in_memory:
                conn.commit()

    def get_pending_merges(self) -> List[Dict[str, Any]]:
        """Return all pending merge proposals."""
        with self._get_conn() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM merge_proposals WHERE status='pending' ORDER BY similarity DESC"
            )
            return [dict(row) for row in cursor.fetchall()]

    def merge_persons(self, keep_id: str, remove_id: str) -> int:
        """
        Merge remove_id into keep_id.
        Returns number of embeddings transferred.
        """
        with self._get_conn() as conn:
            # Move embeddings
            cursor = conn.execute(
                "UPDATE person_embeddings SET person_id=? WHERE person_id=?",
                (keep_id, remove_id),
            )
            moved = cursor.rowcount

            # Move camera history
            conn.execute(
                "UPDATE camera_history SET person_id=? WHERE person_id=?",
                (keep_id, remove_id),
            )

            # Keep earlier first_seen_time
            conn.execute("""
                UPDATE persons
                SET first_seen_time = (
                    SELECT MIN(first_seen_time)
                    FROM persons WHERE person_id IN (?, ?)
                )
                WHERE person_id = ?
            """, (keep_id, remove_id, keep_id))

            # Delete removed person
            conn.execute("DELETE FROM persons WHERE person_id=?", (remove_id,))

            # Update gallery_size on keep_id
            cursor2 = conn.execute(
                "SELECT COUNT(*) FROM person_embeddings WHERE person_id=?", (keep_id,)
            )
            new_size = cursor2.fetchone()[0]
            conn.execute(
                "UPDATE persons SET gallery_size=? WHERE person_id=?",
                (new_size, keep_id),
            )

            # Mark merge proposals as accepted
            conn.execute(
                "UPDATE merge_proposals SET status='accepted', resolved_at=? "
                "WHERE (person_id_a=? AND person_id_b=?) OR (person_id_a=? AND person_id_b=?)",
                (datetime.datetime.now().isoformat(), keep_id, remove_id, remove_id, keep_id),
            )
            if self._in_memory:
                conn.commit()
        return moved

    # ------------------------------------------------------------------
    # Legacy compatibility
    # ------------------------------------------------------------------

    def get_all_embeddings(self) -> List[Tuple[str, np.ndarray]]:
        """Fetch first embedding per person. Prefer get_all_galleries_typed()."""
        with self._get_conn() as conn:
            cursor = conn.execute(
                "SELECT person_id, embedding FROM person_embeddings "
                "GROUP BY person_id"
            )
            rows = cursor.fetchall()
        return [(pid, np.frombuffer(emb, dtype=np.float32)) for pid, emb in rows]
