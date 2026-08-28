#!/usr/bin/env python3
"""
Burnout Prediction API Server
Serves test students and PP2 concurrent data from CSVs via REST API
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import json
from pathlib import Path

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# Global data storage
TEST_STUDENTS_DF = None
PP2_STUDENTS_DF = None
CROSS_YEAR_VALIDATION = None
MODEL_PERFORMANCE = None
SLIIT_ANOMALY_DF = None

# Configuration
CSV_DIR = Path(__file__).parent / "data"  # CSV files should be in ./data folder
TEST_CSV = "model8_predictions.csv"
PP2_CSV = "pp2_final_results.csv"

# SLIIT synthetic-data results (Karunarathne D C — SLIIT BurnoutDetector module)
SLIIT_METRICS_DIR = (
    Path(__file__).parent.parent.parent
    / "sliit-synthetic" / "SLIIT BurnoutDetector" / "results" / "metrics"
)
SLIIT_ANOMALY_CSV = "sliit_anomaly_scores_final.csv"

# NOTE: OULAD behavioral (VAE) risk is no longer served from here.
# It now has its own dedicated API — see vae_api.py in
# sliit-synthetic/Behavioral Anomaly Detection VAE/src/ (port 5003).

def load_data():
    """Load CSV files on startup"""
    global TEST_STUDENTS_DF, PP2_STUDENTS_DF, CROSS_YEAR_VALIDATION, MODEL_PERFORMANCE, SLIIT_ANOMALY_DF

    try:
        print("📂 Loading CSV files...")

        # Load test students
        test_path = CSV_DIR / TEST_CSV
        if test_path.exists():
            TEST_STUDENTS_DF = pd.read_csv(test_path)
            print(f"✅ Test students loaded: {len(TEST_STUDENTS_DF)} students")
        else:
            print(f"⚠️  Test CSV not found at {test_path}")

        # Load PP2 students
        pp2_path = CSV_DIR / PP2_CSV
        if pp2_path.exists():
            PP2_STUDENTS_DF = pd.read_csv(pp2_path)
            # Add derived fields
            PP2_STUDENTS_DF['predictions_agree'] = PP2_STUDENTS_DF['count_based_pred'] == PP2_STUDENTS_DF['max_pool_pred']
            PP2_STUDENTS_DF['alert_level'] = PP2_STUDENTS_DF['actual'].apply(lambda x: 'HIGH' if x == 1 else 'LOW')
            print(f"✅ PP2 students loaded: {len(PP2_STUDENTS_DF)} students")
        else:
            print(f"⚠️  PP2 CSV not found at {pp2_path}")

        # Cross-year validation metrics
        # Loaded from real computed results (cross_year_validation.py) when
        # available; falls back to the last known-good hard-coded snapshot
        # only if that CSV hasn't been generated yet.
        cy_csv_path = CSV_DIR / "cross_year_validation.csv"
        if cy_csv_path.exists():
            cy_df = pd.read_csv(cy_csv_path)
            week17 = cy_df[cy_df['horizon'] == 'Week 17'].iloc[0]
            CROSS_YEAR_VALIDATION = {
                "train_year": int(week17['train_year']),
                "test_year": int(week17['test_year']),
                "metrics": {
                    "f1_score": float(week17['f1_score']),
                    "auc_score": float(week17['auc_score']),
                    "accuracy": float(week17['accuracy']),
                    "precision": float(week17['precision']),
                    "recall": float(week17['recall'])
                },
                "per_horizon": cy_df.to_dict('records'),
                "description": "Model trained on 2013 cohort, tested on 2014 cohort",
                "source": "computed",
                "notes": "Cross-year generalization computed from cross_year_validation.py"
            }
            print(f"CROSS-YEAR validation loaded from CSV ({len(cy_df)} horizons)")
        else:
            CROSS_YEAR_VALIDATION = {
                "train_year": 2013,
                "test_year": 2014,
                "metrics": {
                    "f1_score": 0.7193,
                    "auc_score": 0.8809,
                    "accuracy": 0.82,
                    "precision": 0.78,
                    "recall": 0.75,
                    "balanced_accuracy": 0.81
                },
                "description": "Model trained on 2013 cohort, tested on 2014 cohort",
                "baseline_models_beaten": 7,
                "total_baselines": 7,
                "source": "fallback_hardcoded",
                "notes": "cross_year_validation.csv not found -- run cross_year_validation.py to generate live results"
            }
            print("CROSS-YEAR validation CSV not found, using fallback hard-coded values")

        # Model performance
        MODEL_PERFORMANCE = {
            "accuracy": 0.7227,
            "f1_score": 0.7227,
            "auc_roc": 0.8831,
            "precision": 0.72,
            "recall": 0.72,
            "specificity": 0.72,
            "dataset_info": "Model 8 - OULAD Test Set",
            "total_students": len(TEST_STUDENTS_DF) if TEST_STUDENTS_DF is not None else 0
        }

        # Load SLIIT synthetic-data anomaly scores
        sliit_path = SLIIT_METRICS_DIR / SLIIT_ANOMALY_CSV
        if sliit_path.exists():
            SLIIT_ANOMALY_DF = pd.read_csv(sliit_path)
            print(f"✅ SLIIT synthetic students loaded: "
                  f"{SLIIT_ANOMALY_DF['student_id'].nunique()} students")
        else:
            print(f"⚠️  SLIIT synthetic CSV not found at {sliit_path}")

        print("✅ All data loaded successfully!\n")
        return True

    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "online",
        "test_students": len(TEST_STUDENTS_DF) if TEST_STUDENTS_DF is not None else 0,
        "pp2_students": len(PP2_STUDENTS_DF) if PP2_STUDENTS_DF is not None else 0,
        "api_version": "1.0"
    })

# ============================================================================
# TEST STUDENTS ENDPOINTS
# ============================================================================

@app.route('/api/test-students', methods=['GET'])
def get_test_students():
    """Get all test students with optional filtering"""
    if TEST_STUDENTS_DF is None:
        return jsonify({"error": "Test data not loaded"}), 500

    alert_level = request.args.get('alert_level', None)
    limit = request.args.get('limit', None, type=int)

    df = TEST_STUDENTS_DF.copy()

    # Filter by alert level
    if alert_level and alert_level in ['HIGH', 'MEDIUM', 'LOW']:
        df = df[df['alert_level'] == alert_level]

    # Limit results
    if limit:
        df = df.head(limit)

    # Convert to JSON-friendly format
    students = df.to_dict('records')

    return jsonify({
        "total": len(students),
        "data": students,
        "filters": {
            "alert_level": alert_level,
            "limit": limit
        }
    })

@app.route('/api/test-students/<student_id>', methods=['GET'])
def get_test_student(student_id):
    """Get single test student by ID"""
    if TEST_STUDENTS_DF is None:
        return jsonify({"error": "Test data not loaded"}), 500

    # Try to find by student_id or index
    student = TEST_STUDENTS_DF[TEST_STUDENTS_DF['student_id'] == student_id]

    if student.empty:
        # Try by index
        try:
            idx = int(student_id)
            student = TEST_STUDENTS_DF.iloc[[idx]]
        except:
            return jsonify({"error": "Student not found"}), 404

    if student.empty:
        return jsonify({"error": "Student not found"}), 404

    return jsonify(student.iloc[0].to_dict())

@app.route('/api/test-students/stats', methods=['GET'])
def get_test_stats():
    """Get test students statistics"""
    if TEST_STUDENTS_DF is None:
        return jsonify({"error": "Test data not loaded"}), 500

    stats = {
        "total": len(TEST_STUDENTS_DF),
        "high_risk": len(TEST_STUDENTS_DF[TEST_STUDENTS_DF['alert_level'] == 'HIGH']),
        "medium_risk": len(TEST_STUDENTS_DF[TEST_STUDENTS_DF['alert_level'] == 'MEDIUM']),
        "low_risk": len(TEST_STUDENTS_DF[TEST_STUDENTS_DF['alert_level'] == 'LOW']),
        "actual_dropouts": len(TEST_STUDENTS_DF[TEST_STUDENTS_DF['actual_label'] == 1]),
        "actual_passed": len(TEST_STUDENTS_DF[TEST_STUDENTS_DF['actual_label'] == 0]),
        "avg_risk_score": float(TEST_STUDENTS_DF['academic_risk'].mean())
    }
    return jsonify(stats)

# ============================================================================
# PP2 STUDENTS ENDPOINTS
# ============================================================================

@app.route('/api/pp2-students', methods=['GET'])
def get_pp2_students():
    """Get all PP2 students with optional filtering"""
    if PP2_STUDENTS_DF is None:
        return jsonify({"error": "PP2 data not loaded"}), 500

    alert_level = request.args.get('alert_level', None)
    agreement = request.args.get('agreement', None)  # true/false
    limit = request.args.get('limit', None, type=int)

    df = PP2_STUDENTS_DF.copy()

    # Filter by alert level
    if alert_level and alert_level in ['HIGH', 'LOW']:
        df = df[df['alert_level'] == alert_level]

    # Filter by prediction agreement
    if agreement:
        agree = agreement.lower() == 'true'
        df = df[df['predictions_agree'] == agree]

    # Limit results
    if limit:
        df = df.head(limit)

    # Rename columns for JSON response
    students = []
    for _, row in df.iterrows():
        student = {
            'student_id': f"PP2_{row['id_student']}",
            'n_modules': int(row['n_modules']),
            'actual_label': int(row['actual']),
            'count_based_pred': int(row['count_based_pred']),
            'max_pool_pred': int(row['max_pool_pred']),
            'predictions_agree': bool(row['predictions_agree']),
            'alert_level': row['alert_level']
        }
        students.append(student)

    return jsonify({
        "total": len(students),
        "data": students,
        "filters": {
            "alert_level": alert_level,
            "agreement": agreement,
            "limit": limit
        }
    })

@app.route('/api/pp2-students/<student_id>', methods=['GET'])
def get_pp2_student(student_id):
    """Get single PP2 student by ID"""
    if PP2_STUDENTS_DF is None:
        return jsonify({"error": "PP2 data not loaded"}), 500

    # Handle both "PP2_xxxx" and "xxxx" formats
    if student_id.startswith('PP2_'):
        student_id = student_id[4:]

    try:
        id_num = int(student_id)
    except:
        return jsonify({"error": "Invalid student ID format"}), 400

    student = PP2_STUDENTS_DF[PP2_STUDENTS_DF['id_student'] == id_num]

    if student.empty:
        return jsonify({"error": "Student not found"}), 404

    row = student.iloc[0]
    return jsonify({
        'student_id': f"PP2_{row['id_student']}",
        'n_modules': int(row['n_modules']),
        'actual_label': int(row['actual']),
        'count_based_pred': int(row['count_based_pred']),
        'max_pool_pred': int(row['max_pool_pred']),
        'predictions_agree': bool(row['predictions_agree']),
        'alert_level': row['alert_level']
    })

@app.route('/api/pp2-students/stats', methods=['GET'])
def get_pp2_stats():
    """Get PP2 students statistics"""
    if PP2_STUDENTS_DF is None:
        return jsonify({"error": "PP2 data not loaded"}), 500

    agree = PP2_STUDENTS_DF['predictions_agree'].sum()
    total = len(PP2_STUDENTS_DF)

    stats = {
        "total": total,
        "high_risk": len(PP2_STUDENTS_DF[PP2_STUDENTS_DF['alert_level'] == 'HIGH']),
        "low_risk": len(PP2_STUDENTS_DF[PP2_STUDENTS_DF['alert_level'] == 'LOW']),
        "methods_agree": int(agree),
        "methods_disagree": int(total - agree),
        "agreement_percentage": round(100 * agree / total, 1),
        "modules_1": len(PP2_STUDENTS_DF[PP2_STUDENTS_DF['n_modules'] == 1]),
        "modules_2": len(PP2_STUDENTS_DF[PP2_STUDENTS_DF['n_modules'] == 2]),
        "modules_3": len(PP2_STUDENTS_DF[PP2_STUDENTS_DF['n_modules'] == 3])
    }
    return jsonify(stats)

# ============================================================================
# VALIDATION & METRICS ENDPOINTS
# ============================================================================

@app.route('/api/cross-year-validation', methods=['GET'])
def get_validation():
    """Get cross-year validation results"""
    return jsonify(CROSS_YEAR_VALIDATION)

@app.route('/api/model-performance', methods=['GET'])
def get_model_performance():
    """Get model performance metrics"""
    return jsonify(MODEL_PERFORMANCE)

@app.route('/api/summary', methods=['GET'])
def get_summary():
    """Get combined summary statistics"""
    return jsonify({
        "test_students": get_test_stats().json,
        "pp2_students": get_pp2_stats().json,
        "cross_year_validation": CROSS_YEAR_VALIDATION,
        "model_performance": MODEL_PERFORMANCE,
        "total_combined": (len(TEST_STUDENTS_DF) if TEST_STUDENTS_DF is not None else 0) +
                         (len(PP2_STUDENTS_DF) if PP2_STUDENTS_DF is not None else 0)
    })

# ============================================================================
# SLIIT SYNTHETIC-DATA ENDPOINTS (Karunarathne D C — SLIIT BurnoutDetector)
# ============================================================================

@app.route('/api/sliit-students', methods=['GET'])
def get_sliit_students():
    """Get SLIIT synthetic-cohort behavioral anomaly records, one row per
    student-week. Supports filtering by student_type and module_code."""
    if SLIIT_ANOMALY_DF is None:
        return jsonify({"error": "SLIIT synthetic data not loaded"}), 500

    student_type = request.args.get('student_type', None)
    module_code = request.args.get('module_code', None)
    limit = request.args.get('limit', None, type=int)

    df = SLIIT_ANOMALY_DF.copy()

    if student_type and student_type != 'ALL':
        df = df[df['student_type'] == student_type]
    if module_code and module_code != 'ALL':
        df = df[df['module_code'] == module_code]
    if limit:
        df = df.head(limit)

    return jsonify({
        "total": len(df),
        "unique_students": int(df['student_id'].nunique()),
        "data": df.to_dict('records'),
        "filters": {"student_type": student_type, "module_code": module_code, "limit": limit}
    })

@app.route('/api/sliit-students/<student_id>', methods=['GET'])
def get_sliit_student(student_id):
    """Get all weekly records for a single SLIIT synthetic student."""
    if SLIIT_ANOMALY_DF is None:
        return jsonify({"error": "SLIIT synthetic data not loaded"}), 500

    rows = SLIIT_ANOMALY_DF[
        SLIIT_ANOMALY_DF['student_id'] == student_id
    ].sort_values('week')

    if rows.empty:
        return jsonify({"error": "Student not found"}), 404

    return jsonify({
        "student_id": student_id,
        "student_name": rows['student_name'].iloc[0],
        "student_type": rows['student_type'].iloc[0],
        "at_risk": int(rows['at_risk'].iloc[0]),
        "weekly": rows.to_dict('records')
    })

@app.route('/api/sliit-students/stats', methods=['GET'])
def get_sliit_stats():
    """Summary statistics for the SLIIT synthetic cohort."""
    if SLIIT_ANOMALY_DF is None:
        return jsonify({"error": "SLIIT synthetic data not loaded"}), 500

    per_student = SLIIT_ANOMALY_DF.groupby('student_id').first()
    stats = {
        "total_students": int(per_student.shape[0]),
        "at_risk_students": int(per_student['at_risk'].sum()),
        "student_type_counts": per_student['student_type'].value_counts().to_dict(),
        "modules": sorted(SLIIT_ANOMALY_DF['module_code'].unique().tolist()),
        "avg_anomaly_score": float(SLIIT_ANOMALY_DF['anomaly_score'].mean()),
        "avg_combined_score": float(SLIIT_ANOMALY_DF['combined_score'].mean()),
        "avg_curriculum_compliance": float(SLIIT_ANOMALY_DF['curriculum_compliance'].mean()),
    }
    return jsonify(stats)

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({"error": "Server error"}), 500

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 Burnout Prediction API Server")
    print("=" * 60)
    print()

    if load_data():
        print("🌐 Starting Flask server...")
        print()
        print("📍 API Base URL: http://localhost:5001")
        print()
        print("📚 Available Endpoints:")
        print("   GET  /api/health                    - Health check")
        print("   GET  /api/test-students             - All test students")
        print("   GET  /api/test-students?alert_level=HIGH  - Filtered")
        print("   GET  /api/test-students/<id>        - Single student")
        print("   GET  /api/test-students/stats       - Statistics")
        print("   GET  /api/pp2-students              - All PP2 students")
        print("   GET  /api/pp2-students?agreement=true  - Filtered")
        print("   GET  /api/pp2-students/<id>        - Single student")
        print("   GET  /api/pp2-students/stats       - Statistics")
        print("   GET  /api/cross-year-validation    - Validation results")
        print("   GET  /api/model-performance         - Model metrics")
        print("   GET  /api/summary                   - All combined data")
        print("   GET  /api/sliit-students             - SLIIT synthetic cohort")
        print("   GET  /api/sliit-students?student_type=BURNOUT  - Filtered")
        print("   GET  /api/sliit-students/<id>       - Single SLIIT student")
        print("   GET  /api/sliit-students/stats      - SLIIT cohort stats")
        print()
        print("   ℹ️  Behavioral (VAE) risk is now served separately by vae_api.py")
        print("      (port 5003) — see sliit-synthetic/Behavioral Anomaly Detection VAE/src/")
        print()
        print("💡 To stop: Press Ctrl+C")
        print("=" * 60)
        print()

        app.run(debug=True, host='0.0.0.0', port=5001)
    else:
        print("❌ Failed to load data. Check CSV file paths.")
        print(f"   Looking for CSVs in: {CSV_DIR}")
