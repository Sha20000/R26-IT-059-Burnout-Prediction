# R26-IT-059 — Meta-Integration Layer

**Project:** Explainable Multi-Modal AI System for Early Academic Burnout and Dropout Prediction  
**Component:** Meta-Integration FNN Layer  
**Student:** IT22253194 Jayawickrama G.T.  
**Institute:** SLIIT Faculty of Computing | 2026  

---

## What This Component Does

This component combines risk signals from two specialist models into one final burnout risk score using a Feed-Forward Neural Network (FNN).

```
IT22916426 — GRU Academic Risk     →
                                      Meta-Integration FNN → Final Burnout Risk
IT22215710 — VAE Behavioural Risk  →
```

---

## Folder Structure

```
R26-IT-059-Burnout-Prediction/
│
├── data/
│   ├── academic_prediction.csv       ← from IT22916426 (GRU model)
│   └── behavior_prediction.csv       ← from IT22215710 (VAE model)
│
├── meta_results/                     ← auto-created when you run
│   ├── final_burnout_predictions.csv ← final output
│   ├── meta_fnn_best.pt              ← saved FNN weights
│   ├── meta_scaler.pkl               ← feature scaler
│   └── meta_config.json              ← model config for API
│
├── meta_integration.py               ← train the FNN
├── meta_api.py                       ← Flask REST API
├── index.html                        ← browser dashboard
├── requirements.txt                  ← all dependencies
└── README.md                         ← this file
```

---

## Requirements

- Python 3.10 or higher
- pip (comes with Python)

---

## Part 1 — Install Python on Windows

**Step 1** — Go to https://www.python.org/downloads/windows

**Step 2** — Click the yellow **Download Python 3.x.x** button

**Step 3** — Run the installer. On the first screen **tick the box that says Add Python to PATH** — this is critical

**Step 4** — Click Install Now

**Step 5** — Verify in Command Prompt:
```
python --version
```
You should see `Python 3.x.x`

---

## Part 2 — Install PyTorch on Windows

First create and activate a virtual environment:

```bash
# Navigate to your project folder
cd C:\Users\YourName\Documents\GitHub\R26-IT-059-Burnout-Prediction

# Create virtual environment
python -m venv .venv

# Activate it (do this every time you open a new terminal)
.venv\Scripts\activate
```

Your terminal will show `(.venv)` when active.

### Option A — No GPU (CPU only) — most common for laptops

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Option B — NVIDIA GPU (CUDA 11.8)

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Verify PyTorch installed correctly

```bash
python -c "import torch; print(torch.__version__)"
```

You should see a version number printed.

---

## Part 3 — Install All Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` does not exist install manually:

```bash
pip install torch pandas numpy scikit-learn flask flask-cors
```

> **Note:** Always activate `.venv` before installing. If you see `ModuleNotFoundError` the environment is not active.

---

## Part 4 — Run the System

You need **two terminal windows** open at the same time.

### Step 1 — Put CSVs in the data folder

```
data/academic_prediction.csv    ← copy model8_predictions.csv here and rename
data/behavior_prediction.csv    ← from IT22215710
```

### Step 2 — Train the FNN (run once)

```bash
python meta_integration.py
```

Wait until you see `COMPLETE` printed. This creates the `meta_results/` folder automatically.

### Step 3 — Start the API (Terminal 1 — keep open)

```bash
python meta_api.py
```

You should see:
```
API ready at http://localhost:5002
```

Do **not** close this terminal.

### Step 4 — Open the dashboard (Terminal 2 or File Explorer)

```bash
start index.html
```

Or double click `index.html` in File Explorer.

The dashboard opens in your browser. The header shows:
- 🟢 **API Live** — connected to Flask API, all students loaded
- 🟡 **Static Mode** — API not running, fallback data shown

### Available API routes

| Route | What it returns |
|---|---|
| `http://localhost:5002/health` | API status and model info |
| `http://localhost:5002/students` | All student predictions |
| `http://localhost:5002/students?alert=HIGH` | HIGH risk students only |
| `http://localhost:5002/students/7` | One student by index |
| `http://localhost:5002/summary` | Counts and model performance |

---

## Part 5 — Push to GitHub on Windows

### Step 1 — Install Git

Go to https://git-scm.com/download/win and install with all default settings.

Verify:
```bash
git --version
```

### Step 2 — Configure Git (first time only)

```bash
git config --global user.name "Your Name"
git config --global user.email "your@email.com"
```

### Step 3 — Create .gitignore

Create a file called `.gitignore` in your project root with this content:

```
.venv/
__pycache__/
*.pt
*.pkl
.DS_Store
meta_results/
```

> **Important:** The `.pt` and `.pkl` model files are large. The `meta_results/` folder is auto-generated. Do not push either of these. Only push source code.

### Step 4 — Stage your files

```bash
git add meta_integration.py
git add meta_api.py
git add index.html
git add .gitignore
git add data/academic_prediction.csv
git add requirements.txt
git add README.md
```

### Step 5 — Check what is staged

```bash
git status
```

Files should appear in **green** under "Changes to be committed".

### Step 6 — Commit

```bash
git commit -m "Add meta-integration FNN layer with Flask API and dashboard"
```

### Step 7 — Push

```bash
git push origin feature/academic-trend-IT22916426
```

---

## Common Errors and Fixes

| Error | Fix |
|---|---|
| `ModuleNotFoundError: flask` | Run `pip install flask flask-cors` with `.venv` active |
| `ModuleNotFoundError: torch` | Run PyTorch install command from Part 2 |
| `FileNotFoundError: academic_prediction.csv` | Copy `model8_predictions.csv` to `data/` and rename it |
| `RuntimeError: size mismatch for net.0.weight` | `MetaFNN` in `meta_api.py` does not match training — use `h=32` in both files |
| `FileNotFoundError: meta_fnn_best.pt` | Run `meta_integration.py` before `meta_api.py` |
| `Loss is NaN during training` | VAE features all missing — check `behavior_prediction.csv` loaded correctly |
| `404 on http://localhost:5002/` | Normal — use `/health` `/students` `/summary` routes |
| Dashboard shows Static Mode | `meta_api.py` is not running — start it in a separate terminal |
| `git push` rejected — large file | Add file to `.gitignore` then `git rm --cached filename` |
| `.venv\Scripts\activate` fails in PowerShell | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |

---

## Model Architecture

```
Input (10 features):
  GRU signals (IT22916426):
    academic_risk       — Week 17 overall risk
    week4_risk          — Early horizon
    week8_risk          — Mid horizon
    week12_risk         — Late horizon
    week17_risk         — Optimal horizon

  VAE signals (IT22215710):
    behavior_risk_mean  — Average weekly behaviour risk
    behavior_risk_max   — Peak weekly behaviour risk
    compliance_mean     — Average curriculum compliance
    anomaly_mean        — Average anomaly score
    high_anomaly_weeks  — Count of high anomaly weeks

FNN Architecture:
  Linear(10→32) → ReLU → Dropout(0.3)
  Linear(32→16) → ReLU → Dropout(0.3)
  Linear(16→8)  → ReLU
  Linear(8→1)   → Sigmoid

Output:
  final_burnout_risk (0.0–1.0)
  final_alert (HIGH / MEDIUM / LOW)
```

---

## Alert Levels

| Level | Risk Score | Meaning |
|---|---|---|
| HIGH | > 0.70 | Immediate intervention needed |
| MEDIUM | 0.40–0.70 | Monitor closely |
| LOW | < 0.40 | On track |

---

## Output CSV Columns

The `final_burnout_predictions.csv` contains:

| Column | Description |
|---|---|
| `student_id` | Student identifier |
| `actual_label` | Ground truth (0=safe, 1=at-risk) |
| `academic_risk` | GRU Week 17 risk score |
| `week4/8/12/17_risk` | GRU horizon scores |
| `behavior_risk_mean` | VAE average behaviour risk |
| `compliance_mean` | VAE curriculum compliance |
| `final_burnout_risk` | **Meta-FNN final score** |
| `final_alert` | **HIGH / MEDIUM / LOW** |

---

*R26-IT-059 | SLIIT | 2025–2026*