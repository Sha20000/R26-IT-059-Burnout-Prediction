//Api Configuration

const API_URL = 'http://localhost:5001';
const API_TIMEOUT = 3000;

//Feature Name mapping
const FEATURE_NAMES = {
    'F1_login_count':         'Login Count',
    'F2_vle_clicks':          'VLE Clicks',
    'F3_avg_score':           'Avg Score',
    'F4_num_submitted':       'Submissions',
    'F5_on_time_rate':        'On-Time Rate',
    'F6_inactive_days':       'Inactive Days',
    'F7_score_change':        'Score Change',
    'F8_running_avg':         'Running Avg',
    'F9_score_variance':      'Score Variance',
    'F10_min_score':          'Min Score',
    'F11_max_score':          'Max Score',
    'F12_late_count':         'Late Count',
    'F13_weeks_since_active': 'Weeks Since Active'
  };

//State
let allStudents = [];
let selectedId  = null;
let usingAPI    = false;

//API Helpers 
async function fetchWithTimeout(url, ms = API_TIMEOUT) {
    const controller = new AbortController();
    const timer      = setTimeout(() => controller.abort(), ms);
    try {
      const res = await fetch(url, { signal: controller.signal });
      clearTimeout(timer);
      return res;
    } catch (err) {
      clearTimeout(timer);
      throw err;
    }
  }

//Load Students
async function loadStudents() {
    setApiStatus('connecting');
  
    try {
      // Try API first
      const res  = await fetchWithTimeout(`${API_URL}/students`);
      const data = await res.json();
  
      if (data.students && data.students.length > 0) {
        allStudents = data.students;
        usingAPI    = true;
        setApiStatus('live');
        document.getElementById('dataSource').textContent = 'Live API';
        console.log(`Loaded ${allStudents.length} students from Flask API`);
  
        // Also load summary from API
        loadSummaryFromAPI();
      } else {
        throw new Error('Empty response from API');
      }
  
    } catch (err) {
      // Fallback to static data.js
      console.warn('API not available — using static data:', err.message);
      usingAPI    = false;
      allStudents = typeof STUDENTS !== 'undefined' ? STUDENTS : [];
      setApiStatus('offline');
      document.getElementById('dataSource').textContent = 'Static';
  
      if (allStudents.length === 0) {
        document.getElementById('loadingMsg').textContent =
          'No data available. Run the Flask API or check data.js';
        return;
      }
    }


//Hide Loading,build UI
document.getElementById('loadingMsg').style.display = 'none';
buildStudentList(allStudents);
updateSummary();

 // Auto-select first HIGH risk student
 const firstHigh = allStudents.find(s => s.alert_level === 'HIGH');
 if (firstHigh) selectStudent(firstHigh);

}

// Load summary stats from API
async function loadSummaryFromAPI() {
    try {
      const res  = await fetchWithTimeout(`${API_URL}/summary`);
      const data = await res.json();
  
      document.getElementById('footerF1').textContent  = data.model_f1?.toFixed(4) || '0.7227';
      document.getElementById('footerAuc').textContent = data.model_auc?.toFixed(4) || '0.8831';
    } catch (err) {
      // Keep default values if summary fails
    }
  }


//API status indicator
function setApiStatus(state) {
    const el = document.getElementById('apiStatus');
    if (state === 'live') {
      el.textContent = '🟢 API Live';
      el.style.color = '#6EE7B7';
    } else if (state === 'offline') {
      el.textContent = '🟡 Static Mode';
      el.style.color = '#FDE68A';
    } else {
      el.textContent = '⏳ Connecting...';
      el.style.color = '#9BA8B7';
    }
  }

