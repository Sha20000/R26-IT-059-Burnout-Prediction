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
