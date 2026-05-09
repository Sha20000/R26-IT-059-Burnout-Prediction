# R26-IT-059-Burnout-Prediction
Explainable Multi-Modal AI for Academic Burnout Prediction SLIIT Final Year Research R26-IT-059

## Overview
This prototype provides a centralized Flask dashboard that loads model outputs from CSV files,
applies weighted meta-integration, enforces override rules for high-risk students, and generates
human-readable explanations with advisor recommendations.

## Quick Start
1. Create a virtual environment and install dependencies.
2. Run the Flask app.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

The dashboard will be available at http://127.0.0.1:5000

## Configuration
Edit the configuration in [config/weights.json](config/weights.json):
- `weights` for model score blending
- `thresholds` for low/medium/high mapping
- `override` rule for automatic high-risk escalation
- `sources` for CSV paths and column mappings

## Data Sources
The prototype uses:
- [model8_predictions.csv](model8_predictions.csv) as the academic model output
- [data/emotion_predictions.csv](data/emotion_predictions.csv)
- [data/behavior_predictions.csv](data/behavior_predictions.csv)
- [data/engagement_predictions.csv](data/engagement_predictions.csv)

Replace or extend the CSV files as needed. Each source maps `student_id` to a score column.

## API Endpoints
- `/api/students` returns the merged records with final score and explanations
- `/api/students/<student_id>` returns a single student record
- `/api/export/csv` downloads the dashboard data as CSV

## Architecture
- `app/services/ingestion.py` loads and merges model outputs
- `app/services/integration.py` applies weighted averaging and override logic
- `app/services/explain.py` generates rule-based explanations
- `app/routes` exposes API and UI routes
- `app/templates/dashboard.html` renders the dashboard

## Notes
- PDF export is handled via the browser print-to-PDF flow.
- Adjust `refresh_seconds` in the config to control reload behavior.
