# 🚀 Flask Backend Setup Guide

**Converting from Static Data (data.js) to API-Driven Architecture**

---

## 📋 Overview

Instead of baking data into JavaScript, you now have:
- **Single CSV source of truth** (model8_predictions.csv + pp2_final_results.csv)
- **Flask REST API** that serves data as JSON
- **Dynamic frontend** that fetches from API
- **Easy updates** - just replace CSV files, restart server

---

## 🛠️ Setup (5 minutes)

### Step 1: Install Dependencies

```bash
pip install flask flask-cors pandas
```

### Step 2: Create Folder Structure

```
burnout-demo/
├── app_api.py              # Flask server (new)
├── api-index.html          # Frontend (new)
├── data/                   # Create this folder
│   ├── model8_predictions.csv
│   └── pp2_final_results.csv
├── index.html              # Old static version (keep as backup)
├── style.css               # Keep existing
└── app.js                  # Keep existing
```

### Step 3: Copy Files

1. Copy **app_api.py** to your project root (where index.html is)
2. Copy **api-index.html** to your project root
3. Create a **data/** folder
4. Move CSV files into **data/** folder:
   ```bash
   mkdir data
   cp model8_predictions.csv data/
   cp pp2_final_results.csv data/
   ```

### Step 4: Start Flask Server

```bash
cd burnout-demo
python app_api.py
```

You should see:
```
============================================================
🚀 Burnout Prediction API Server
============================================================

📂 Loading CSV files...
✅ Test students loaded: 2543 students
✅ PP2 students loaded: 839 students
✅ All data loaded successfully!

🌐 Starting Flask server...

📍 API Base URL: http://localhost:5001

📚 Available Endpoints:
   GET  /api/health                    - Health check
   GET  /api/test-students             - All test students
   GET  /api/test-students?alert_level=HIGH  - Filtered
   GET  /api/test-students/<id>        - Single student
   GET  /api/test-students/stats       - Statistics
   GET  /api/pp2-students              - All PP2 students
   GET  /api/pp2-students?agreement=true  - Filtered
   GET  /api/pp2-students/<id>        - Single student
   GET  /api/pp2-students/stats       - Statistics
   GET  /api/cross-year-validation    - Validation results
   GET  /api/model-performance         - Model metrics
   GET  /api/summary                   - All combined data

💡 To stop: Press Ctrl+C
============================================================
```

### Step 5: Open Frontend

Open **api-index.html** in browser:
```bash
open api-index.html
# or open http://localhost:5001 after opening index.html
```

You should see:
- ✅ "API Live" status in top right
- Students loading from API
- Filter and search working

---

## 📊 API Endpoints Reference

### Health Check
```bash
curl http://localhost:5001/api/health
```

**Response:**
```json
{
  "status": "online",
  "test_students": 2543,
  "pp2_students": 839,
  "api_version": "1.0"
}
```

### Get All Test Students
```bash
curl http://localhost:5001/api/test-students
```

### Get Filtered Test Students
```bash
# By alert level
curl http://localhost:5001/api/test-students?alert_level=HIGH
curl http://localhost:5001/api/test-students?alert_level=LOW
curl http://localhost:5001/api/test-students?alert_level=MEDIUM

# Limit results
curl http://localhost:5001/api/test-students?limit=10
```

### Get Single Test Student
```bash
curl http://localhost:5001/api/test-students/OULAD_TEST_0007
curl http://localhost:5001/api/test-students/0  # by index
```

### Get Test Students Stats
```bash
curl http://localhost:5001/api/test-students/stats
```

**Response:**
```json
{
  "total": 2543,
  "high_risk": 412,
  "medium_risk": 598,
  "low_risk": 1533,
  "actual_dropouts": 478,
  "actual_passed": 2065,
  "avg_risk_score": 0.385
}
```

### Get All PP2 Students
```bash
curl http://localhost:5001/api/pp2-students
```

### Get Filtered PP2 Students
```bash
# By alert level
curl http://localhost:5001/api/pp2-students?alert_level=HIGH
curl http://localhost:5001/api/pp2-students?alert_level=LOW

# By prediction agreement
curl http://localhost:5001/api/pp2-students?agreement=true
curl http://localhost:5001/api/pp2-students?agreement=false

# Combined filters
curl http://localhost:5001/api/pp2-students?alert_level=HIGH&agreement=false&limit=5
```

### Get Single PP2 Student
```bash
curl http://localhost:5001/api/pp2-students/29820
curl http://localhost:5001/api/pp2-students/PP2_29820
```

### Get PP2 Students Stats
```bash
curl http://localhost:5001/api/pp2-students/stats
```

**Response:**
```json
{
  "total": 839,
  "high_risk": 431,
  "low_risk": 408,
  "methods_agree": 585,
  "methods_disagree": 254,
  "agreement_percentage": 69.7,
  "modules_1": 78,
  "modules_2": 750,
  "modules_3": 11
}
```

### Get Cross-Year Validation
```bash
curl http://localhost:5001/api/cross-year-validation
```

### Get Model Performance
```bash
curl http://localhost:5001/api/model-performance
```

### Get Complete Summary
```bash
curl http://localhost:5001/api/summary
```

---

## 🔄 Updating Data

**Without restarting server:**

1. Replace CSV files in `data/` folder
2. Refresh browser (data will reload from API on page load)

**Old approach (bundled data.js):**
- Need to regenerate JavaScript file
- Need to update both data.js and HTML
- Harder to maintain

**New approach (Flask API):**
- Just update CSV files
- Frontend fetches fresh data automatically
- Single source of truth

---

## 🛡️ Troubleshooting

### Issue: "API Offline" in browser
**Solution:** Make sure Flask server is running
```bash
python app_api.py
# Should show "🟢 API Live" status
```

### Issue: "CSV not found"
**Solution:** Check folder structure
```bash
ls data/model8_predictions.csv
ls data/pp2_final_results.csv
# Both files should exist
```

### Issue: CORS errors in browser console
**Solution:** Flask app has CORS enabled (see `from flask_cors import CORS`)
- Restart Flask if you're getting errors
- Check that port 5001 is not in use

### Issue: Port 5001 already in use
**Solution:** Change port in app_api.py
```python
# Line at bottom: change port from 5001 to 5002
app.run(debug=True, host='0.0.0.0', port=5002)
```

---

## 📈 Performance Comparison

| Metric | Static (data.js) | API (Flask) |
|--------|-----------------|------------|
| Initial Load | 139KB bundle | Small HTML + API calls |
| Updates | Regenerate JS | Update CSV files |
| Scalability | Fixed size | Scales easily |
| Offline | Works offline | Needs backend |
| Response Time | Instant | ~50-200ms |
| Backend | None | Flask server |

---

## 🚢 Deployment Options

### Option 1: Local Development (Current)
```bash
python app_api.py
open api-index.html
```
✅ Works offline, quick development

### Option 2: Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app_api.py"]
```

### Option 3: Cloud Deployment (Heroku/AWS/GCP)
- Push app_api.py + data/ folder to cloud
- Flask runs on cloud server
- Frontend fetches from cloud API
- Data updates on cloud storage

---

## ✅ Verification Checklist

- [ ] Flask server running (check "🟢 API Live" status)
- [ ] CSV files in `data/` folder
- [ ] api-index.html opens in browser
- [ ] Student list loads (not empty)
- [ ] Clicking student shows details
- [ ] Filter dropdown works
- [ ] Footer shows correct counts
- [ ] API endpoints respond to curl commands

---

## 📚 Next Steps

1. **Test the API locally** (confirm everything works)
2. **Keep data.js as backup** (in case you need pure static version)
3. **Consider adding database** later (replace CSV with PostgreSQL/MySQL)
4. **Deploy to cloud** when ready for production

---

## 🆘 Need Help?

1. Check Flask console output for errors
2. Open browser DevTools (F12) → Network tab → see API calls
3. Test endpoints with curl before debugging frontend
4. Verify CSV files are in correct location

---

**Created:** August 24, 2026  
**For:** R26-IT-059 Burnout Prediction Project  
**Status:** Production Ready
