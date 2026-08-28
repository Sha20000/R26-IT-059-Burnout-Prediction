#!/usr/bin/env python3
"""
Behavioral Anomaly Detection (VAE) API Server
Serves per-student behavioral risk from behavior_prediction.csv
R26-IT-059 | IT22215710 Karunarathne D C
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
from pathlib import Path

app = Flask(__name__)
CORS(app)

BEHAVIOR_DF = None
CSV_PATH = Path(__file__).parent.parent / "results" / "metrics" / "behavior_prediction.csv"


def load_data():
    global BEHAVIOR_DF
    if CSV_PATH.exists():
        BEHAVIOR_DF = pd.read_csv(CSV_PATH)
        print(f"✅ Behavioral (VAE) records loaded: "
              f"{BEHAVIOR_DF['student_id'].nunique()} students, "
              f"{len(BEHAVIOR_DF):,} rows")
        return True
    else:
        print(f"⚠️  Behavior CSV not found at {CSV_PATH}")
        return False


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "online",
        "students_loaded": int(BEHAVIOR_DF['student_id'].nunique()) if BEHAVIOR_DF is not None else 0,
        "source_csv": str(CSV_PATH.name),
    })


@app.route('/api/behavioral-risk', methods=['GET'])
def get_behavioral_risk():
    """Per-student behavioral risk, aggregated from the weekly
    behavior_prediction.csv (mean / max / mean / mean / sum)."""
    if BEHAVIOR_DF is None:
        return jsonify({"error": "Behavioral VAE data not loaded"}), 500

    agg = BEHAVIOR_DF.groupby('student_id').agg(
        behavior_risk_mean=('behavioral_risk_score', 'mean'),
        behavior_risk_max=('behavioral_risk_score', 'max'),
        compliance_mean=('curriculum_compliance', 'mean'),
        anomaly_mean=('anomaly_score', 'mean'),
        high_anomaly_weeks=('high_anomaly_flag', 'sum'),
    ).reset_index()

    risk_level = request.args.get('risk_level', None)
    if risk_level and risk_level != 'ALL':
        if risk_level == 'HIGH':
            agg = agg[agg['behavior_risk_mean'] > 0.7]
        elif risk_level == 'MEDIUM':
            agg = agg[(agg['behavior_risk_mean'] > 0.4) & (agg['behavior_risk_mean'] <= 0.7)]
        elif risk_level == 'LOW':
            agg = agg[agg['behavior_risk_mean'] <= 0.4]

    return jsonify({
        "total": len(agg),
        "students": agg.to_dict('records')
    })


@app.route('/api/behavioral-risk/<student_id>', methods=['GET'])
def get_behavioral_risk_student(student_id):
    """Full weekly history for a single student."""
    if BEHAVIOR_DF is None:
        return jsonify({"error": "Behavioral VAE data not loaded"}), 500

    try:
        sid = int(student_id)
    except ValueError:
        sid = student_id

    rows = BEHAVIOR_DF[BEHAVIOR_DF['student_id'] == sid].sort_values('week')
    if rows.empty:
        return jsonify({"error": "Student not found"}), 404

    return jsonify({
        "student_id": student_id,
        "weekly": rows.to_dict('records')
    })


@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def server_error(error):
    return jsonify({"error": "Server error"}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("🧠 Behavioral Anomaly Detection (VAE) API Server")
    print("=" * 60)
    if load_data():
        print()
        print("📍 API Base URL: http://localhost:5003")
        print()
        print("📚 Available Endpoints:")
        print("   GET  /api/health                       - Health check")
        print("   GET  /api/behavioral-risk               - All students (aggregated)")
        print("   GET  /api/behavioral-risk?risk_level=HIGH  - Filtered")
        print("   GET  /api/behavioral-risk/<student_id>  - Single student, weekly history")
        print()
        print("💡 To stop: Press Ctrl+C")
        print("=" * 60)
        app.run(debug=True, host='0.0.0.0', port=5003)
    else:
        print("❌ Failed to load data. Check that behavior_prediction.csv exists at:")
        print(f"   {CSV_PATH}")
