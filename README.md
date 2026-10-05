# Meta-XAI Student Success Workspace

Meta-XAI is the IT22253194 component of the burnout research project. It does not train another burnout predictor. It turns specialist model outputs into validated evidence, explainable intervention priority, advisor actions, and follow-up workflow.

## Current build

The current UI runs in clearly labelled **DEMO** mode using deterministic OULAD-style fixture profiles. It demonstrates:

- Intervention queue ranked by P1/P2/P3 priority
- Academic trajectory explanation across Weeks 4, 8, 12, and 17
- Evidence coverage and confidence
- Recommended action, owner, and due date
- Student search and queue filters
- Case status updates
- CSV export
- Trainable intervention-opportunity model with temporal evaluation

## Run locally

Install Python 3.11+ and then run:

```powershell
python -m venv .venv
.venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Open <http://127.0.0.1:5000>.

## Data transition

The demo does not claim that the current academic, behavioural, and emotional branch artifacts are aligned. The final mode will consume new outputs generated from the same OULAD cohort and canonical student IDs. See [META_XAI_BUILD_PLAN.md](META_XAI_BUILD_PLAN.md) and [data/incoming/README.md](data/incoming/README.md).

## API

- `GET /api/overview`
- `GET /api/students`
- `GET /api/students?q=...&priority=P1&status=New`
- `GET /api/students/<student_id>`
- `PATCH /api/students/<student_id>/case`
- `GET /api/export/csv`
- `POST /api/analyze` for real-time Meta-XAI analysis

## The trainable model

Meta-XAI includes an **intervention-opportunity model**, not another burnout-risk model. It learns from historical intervention outcomes whether a student profile is likely to benefit from timely support. The model uses the upstream academic trajectory and behavioural/emotional evidence as inputs, with `intervention_success` as an independently recorded target.

Training requires outcome-labelled data. No fabricated or dashboard-demo records are used for training. See [data/training/README.md](data/training/README.md) for the schema and command.
