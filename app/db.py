import json
import logging
from pathlib import Path
import sqlite3
from typing import Any

logger = logging.getLogger(__name__)


def get_connection(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    return conn


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = get_connection(db_path)
    try:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS students (
            student_id TEXT PRIMARY KEY,
            canonical_student_id TEXT,
            university_id TEXT,
            student_name TEXT,
            programme TEXT,
            cohort TEXT,
            enrolled_modules TEXT,
            data_mode TEXT DEFAULT 'OULAD_SYNTHETIC',
            priority_level TEXT,
            effective_priority TEXT,
            priority_score REAL,
            confidence TEXT,
            trajectory TEXT,
            trajectory_explanation TEXT,
            academic_risks TEXT,
            behavior_risk REAL,
            emotional_stress INTEGER,
            available_sources INTEGER,
            expected_sources INTEGER,
            evidence_coverage REAL,
            recommended_action TEXT,
            owner_role TEXT,
            due_date TEXT,
            status TEXT DEFAULT 'New',
            follow_up TEXT DEFAULT 'Pending',
            model_evidence TEXT,
            intervention_opportunity TEXT,
            reason_codes TEXT,
            is_multi_modal_override INTEGER DEFAULT 0,
            is_within_capacity INTEGER DEFAULT 1,
            advisor_notes TEXT DEFAULT '',
            intervention_outcome TEXT DEFAULT '',
            resolved_at TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS intervention_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            canonical_student_id TEXT,
            resolved_at TEXT,
            intervention_outcome TEXT,
            actual_label INTEGER,
            advisor_notes TEXT,
            priority_level TEXT,
            effective_priority TEXT,
            predicted_success_probability REAL,
            academic_risks TEXT,
            emotional_stress INTEGER,
            behavior_risk REAL,
            routed_to_retraining INTEGER DEFAULT 1,
            logged_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS advisor_settings (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)

        # Migration helper for dynamically adding columns if table was created earlier
        existing_cols = {col[1] for col in conn.execute("PRAGMA table_info(students)").fetchall()}
        if "university_id" not in existing_cols:
            conn.execute("ALTER TABLE students ADD COLUMN university_id TEXT;")
        if "enrolled_modules" not in existing_cols:
            conn.execute("ALTER TABLE students ADD COLUMN enrolled_modules TEXT;")

        # Create indexes after ensuring columns exist
        conn.executescript("""
        CREATE INDEX IF NOT EXISTS idx_students_priority ON students (priority_level);
        CREATE INDEX IF NOT EXISTS idx_students_effective_priority ON students (effective_priority);
        CREATE INDEX IF NOT EXISTS idx_students_status ON students (status);
        CREATE INDEX IF NOT EXISTS idx_students_uni_id ON students (university_id);
        CREATE INDEX IF NOT EXISTS idx_logs_student_id ON intervention_logs (student_id);
        """)
        conn.commit()
    finally:
        conn.close()


def _row_to_student_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    for json_col in ("academic_risks", "model_evidence", "intervention_opportunity", "reason_codes", "enrolled_modules"):
        val = d.get(json_col)
        if isinstance(val, str) and val:
            try:
                d[json_col] = json.loads(val)
            except Exception:
                pass
        elif val is None and json_col in ("reason_codes", "academic_risks", "enrolled_modules"):
            d[json_col] = []

    d["is_multi_modal_override"] = bool(d.get("is_multi_modal_override", 0))
    d["is_within_capacity"] = bool(d.get("is_within_capacity", 1))
    return d


def save_or_update_students(records: list[dict], db_path: Path) -> None:
    if not records:
        return
    conn = get_connection(db_path)
    try:
        for r in records:
            enrolled_val = r.get("enrolled_modules")
            if isinstance(enrolled_val, (list, dict)):
                enrolled_str = json.dumps(enrolled_val)
            elif isinstance(enrolled_val, str):
                enrolled_str = enrolled_val
            else:
                enrolled_str = "[]"

            conn.execute("""
            INSERT INTO students (
                student_id, canonical_student_id, university_id, student_name, programme, cohort, enrolled_modules, data_mode,
                priority_level, effective_priority, priority_score, confidence, trajectory,
                trajectory_explanation, academic_risks, behavior_risk, emotional_stress,
                available_sources, expected_sources, evidence_coverage, recommended_action,
                owner_role, due_date, status, follow_up, model_evidence, intervention_opportunity,
                reason_codes, is_multi_modal_override, is_within_capacity, advisor_notes,
                intervention_outcome, resolved_at, updated_at
            ) VALUES (
                :student_id, :canonical_student_id, :university_id, :student_name, :programme, :cohort, :enrolled_modules, :data_mode,
                :priority_level, :effective_priority, :priority_score, :confidence, :trajectory,
                :trajectory_explanation, :academic_risks, :behavior_risk, :emotional_stress,
                :available_sources, :expected_sources, :evidence_coverage, :recommended_action,
                :owner_role, :due_date, :status, :follow_up, :model_evidence, :intervention_opportunity,
                :reason_codes, :is_multi_modal_override, :is_within_capacity, :advisor_notes,
                :intervention_outcome, :resolved_at, CURRENT_TIMESTAMP
            )
            ON CONFLICT(student_id) DO UPDATE SET
                canonical_student_id=excluded.canonical_student_id,
                university_id=excluded.university_id,
                student_name=excluded.student_name,
                programme=excluded.programme,
                cohort=excluded.cohort,
                enrolled_modules=excluded.enrolled_modules,
                priority_level=excluded.priority_level,
                effective_priority=excluded.effective_priority,
                priority_score=excluded.priority_score,
                confidence=excluded.confidence,
                trajectory=excluded.trajectory,
                trajectory_explanation=excluded.trajectory_explanation,
                academic_risks=excluded.academic_risks,
                behavior_risk=excluded.behavior_risk,
                emotional_stress=excluded.emotional_stress,
                available_sources=excluded.available_sources,
                expected_sources=excluded.expected_sources,
                evidence_coverage=excluded.evidence_coverage,
                recommended_action=excluded.recommended_action,
                owner_role=excluded.owner_role,
                due_date=excluded.due_date,
                status=excluded.status,
                follow_up=excluded.follow_up,
                model_evidence=excluded.model_evidence,
                intervention_opportunity=excluded.intervention_opportunity,
                reason_codes=excluded.reason_codes,
                is_multi_modal_override=excluded.is_multi_modal_override,
                is_within_capacity=excluded.is_within_capacity,
                advisor_notes=excluded.advisor_notes,
                intervention_outcome=excluded.intervention_outcome,
                resolved_at=excluded.resolved_at,
                updated_at=CURRENT_TIMESTAMP
            """, {
                "student_id": r.get("student_id"),
                "canonical_student_id": r.get("canonical_student_id", r.get("student_id")),
                "university_id": r.get("university_id", f"IT22{str(r.get('student_id')).split('_')[-1]}"),
                "student_name": r.get("student_name", "Anonymous Student"),
                "enrolled_modules": enrolled_str,
                "programme": r.get("programme", "BSc (Hons) in Information Technology"),
                "cohort": r.get("cohort", "Year 3 Semester 2 (Batch 2022)"),
                "data_mode": r.get("data_mode", "SQLITE DATABASE (OULAD)"),
                "priority_level": r.get("priority_level", "P3"),
                "effective_priority": r.get("effective_priority", r.get("priority_level", "P3")),
                "priority_score": _safe_float(r.get("priority_score"), 0.0),
                "confidence": str(r.get("confidence", "Medium")),
                "trajectory": str(r.get("trajectory", "Stable")),
                "trajectory_explanation": str(r.get("trajectory_explanation", "")),
                "academic_risks": json.dumps(r.get("academic_risks", [])),
                "behavior_risk": _safe_float(r.get("behavior_risk", r.get("behaviour_risk")), 0.0),
                "emotional_stress": _safe_int(r.get("emotional_stress"), 0),
                "available_sources": _safe_int(r.get("available_sources"), 3),
                "expected_sources": _safe_int(r.get("expected_sources"), 3),
                "evidence_coverage": _safe_float(r.get("evidence_coverage"), 1.0),
                "recommended_action": str(r.get("recommended_action", "Monitor")),
                "owner_role": str(r.get("owner_role", "Programme advisor")),
                "due_date": str(r.get("due_date", "")),
                "status": str(r.get("status", "New")),
                "follow_up": str(r.get("follow_up", "Pending")),
                "model_evidence": json.dumps(r.get("model_evidence", {})),
                "intervention_opportunity": json.dumps(r.get("intervention_opportunity", {})),
                "reason_codes": json.dumps(r.get("reason_codes", [])),
                "is_multi_modal_override": 1 if r.get("is_multi_modal_override") else 0,
                "is_within_capacity": 1 if r.get("is_within_capacity", True) else 0,
                "advisor_notes": str(r.get("advisor_notes", "")),
                "intervention_outcome": str(r.get("intervention_outcome", "")),
                "resolved_at": str(r.get("resolved_at", "")),
            })
        conn.commit()
    finally:
        conn.close()


def fetch_all_students(db_path: Path) -> list[dict]:
    conn = get_connection(db_path)
    try:
        rows = conn.execute("SELECT * FROM students ORDER BY priority_score DESC").fetchall()
        return [_row_to_student_dict(row) for row in rows]
    finally:
        conn.close()


def fetch_student(student_id: str, db_path: Path) -> dict | None:
    conn = get_connection(db_path)
    try:
        row = conn.execute("SELECT * FROM students WHERE student_id = ? OR university_id = ?", (student_id, student_id)).fetchone()
        return _row_to_student_dict(row) if row else None
    finally:
        conn.close()


def update_case_in_db(student_id: str, payload: dict, db_path: Path) -> dict | None:
    conn = get_connection(db_path)
    try:
        row = conn.execute("SELECT * FROM students WHERE student_id = ? OR university_id = ?", (student_id, student_id)).fetchone()
        if not row:
            return None
        student = _row_to_student_dict(row)
        actual_student_id = student["student_id"]

        status = payload.get("status", "Resolved")
        follow_up = payload.get("follow_up", student.get("follow_up", "Scheduled"))
        intervention_outcome = payload.get("intervention_outcome", student.get("intervention_outcome", ""))
        advisor_notes = payload.get("notes", payload.get("advisor_notes", student.get("advisor_notes", "")))
        from datetime import date
        resolved_at = str(date.today())

        conn.execute("""
            UPDATE students SET
                status = ?,
                follow_up = ?,
                intervention_outcome = ?,
                advisor_notes = ?,
                resolved_at = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE student_id = ?
        """, (status, follow_up, intervention_outcome, advisor_notes, resolved_at, actual_student_id))

        outcome_str = str(intervention_outcome)
        actual_label = 1 if "success" in outcome_str.lower() or "improved" in outcome_str.lower() else 0

        opp = student.get("intervention_opportunity") or {}
        pred_prob = _safe_float(opp.get("intervention_success_probability"), 0.0)

        conn.execute("""
            INSERT INTO intervention_logs (
                student_id, canonical_student_id, resolved_at, intervention_outcome,
                actual_label, advisor_notes, priority_level, effective_priority,
                predicted_success_probability, academic_risks, emotional_stress,
                behavior_risk, routed_to_retraining
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
        """, (
            actual_student_id,
            student.get("canonical_student_id"),
            resolved_at,
            outcome_str,
            actual_label,
            advisor_notes,
            student.get("priority_level"),
            student.get("effective_priority"),
            pred_prob,
            json.dumps(student.get("academic_risks", [])),
            _safe_int(student.get("emotional_stress"), 0),
            _safe_float(student.get("behavior_risk"), 0.0),
        ))
        conn.commit()

        updated_row = conn.execute("SELECT * FROM students WHERE student_id = ?", (actual_student_id,)).fetchone()
        return _row_to_student_dict(updated_row) if updated_row else None
    finally:
        conn.close()


def clear_students_in_db(db_path: Path) -> None:
    conn = get_connection(db_path)
    try:
        conn.execute("DELETE FROM students;")
        conn.commit()
    finally:
        conn.close()


def fetch_intervention_logs(db_path: Path) -> list[dict]:
    conn = get_connection(db_path)
    try:
        rows = conn.execute("SELECT * FROM intervention_logs ORDER BY logged_at DESC").fetchall()
        res = []
        for r in rows:
            d = dict(r)
            if d.get("academic_risks"):
                try:
                    d["academic_risks"] = json.loads(d["academic_risks"])
                except Exception:
                    pass
            res.append(d)
        return res
    finally:
        conn.close()


def get_db_stats(db_path: Path) -> dict[str, Any]:
    if not db_path.exists():
        return {"exists": False, "total_students": 0, "total_intervention_audit_logs": 0}
    conn = get_connection(db_path)
    try:
        st_count = conn.execute("SELECT COUNT(*) FROM students").fetchone()[0]
        log_count = conn.execute("SELECT COUNT(*) FROM intervention_logs").fetchone()[0]
        resolved_count = conn.execute("SELECT COUNT(*) FROM students WHERE status = 'Resolved'").fetchone()[0]
        p1_count = conn.execute("SELECT COUNT(*) FROM students WHERE priority_level = 'P1'").fetchone()[0]
        file_size_kb = round(db_path.stat().st_size / 1024, 2)
        return {
            "exists": True,
            "database_file": db_path.name,
            "file_size_kb": file_size_kb,
            "total_students": st_count,
            "total_intervention_audit_logs": log_count,
            "resolved_cases": resolved_count,
            "p1_cases": p1_count,
        }
    finally:
        conn.close()


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val) if val is not None else default
    except (ValueError, TypeError):
        return default


def _safe_int(val: Any, default: int = 0) -> int:
    try:
        return int(val) if val is not None else default
    except (ValueError, TypeError):
        return default
