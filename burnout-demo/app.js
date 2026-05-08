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


//Risk colour
function riskColor(value) {
    if (value >= 0.7) return '#C00000';
    if (value >= 0.4) return '#B86B00';
    return '#2E7D32';
  }  


  //Advisor text

function getAdvisorText(student){

    const pct = (student.academic_risk * 100).toFixed(0);
    const f1 = FEATURE_NAMES[student.top_feature_1] || student.top_feature_1;

    if(student.alert_level === 'HIGH'){
        return `This student shows a ${pct}% burnout/dropout risk at Week 17. ` +
      `Immediate intervention is recommended. ` +
      `The model identifies ${f1} as the strongest signal. ` +
      `Schedule a welfare meeting within the next week and review VLE engagement urgently.`;
    
    } else if(student.alert_level === 'MEDIUM'){
        return `This student shows a ${pct}% risk level — monitor closely over the next 2–3 weeks. ` +
      `Their ${f1} pattern requires attention. ` +
      `Consider a check-in meeting and encourage engagement with VLE resources.`;
    } else {
        return `Risk level is LOW (${pct}%). This student appears to be on track. ` +
        `Continue routine monitoring. ` +
        `The model's primary signal is ${f1}.`;
    }


  }


//Build student sidebar list

function buildStudentList(students){

    const list = document.getElementById('studentList');
    list.innerHTML = '';

    if(students.length === 0){
        list.innerHTML = '<div style="padding:16px;color:#999;font-size:13px">No students found</div>';
        return;
    }

    students.forEach(s => {

        const row = document.createElement('div');
        row.className = 'student-row' + (s.student_id === selectedId ? 'selected' : '');
        row.dataset.id = s.student_id;

        const shortId = s.student_id.replace('OULAD_TEST_','Student #');
        const riskPct = (s.academic_risk * 100).toFixed(1);

        row.innerHTML = `
      <div class="student-row-left">
        <span class="student-row-id">${shortId}</span>
        <span class="student-row-sub">Risk: ${riskPct}%</span>
      </div>
      <span class="alert-pill ${s.alert_level}">${s.alert_level}</span>
    `;

    row.addEventListener('click', () => selectStudent(s));
    list.appendChild(row);





    });

}    

//Filter dropdown
async function filterStudents() {

    const level = document.getElementById('filterAlert').value;

    if(usingAPI){
        try{
            const url = level === 'ALL'
            ? `${API_URL}/students`
        : `${API_URL}/students?alert=${level}`;
        const res = await fetchWithTimeout(url);
        const data = await res.json();
        buildStudentList(data.students || []);
        return;

        }catch (err){
            console.warn('API filter failed, falling back to static filter:', err.message);

        }
    }

    //local filter fallback
    const filtered = level === 'ALL'
    ? allStudents
    : allStudents.filter(s => s.alert_level === level);
  
    buildStudentList(filtered);
    
}

//Summary footer stats
function updateSummary(){

    document.getElementById('totalStudents').textContent = allStudents.length;
    document.getElementById('highCount').textContent = allStudents.filter(s => s.alert_level === 'HIGH').length;
    document.getElementById('mediumCount').textContent = allStudents.filter(s => s.alert_level === 'MEDIUM').length;
    document.getElementById('lowCount').textContent = allStudents.filter(s => s.alert_level === 'LOW').length;



}

//Select and render student details

async function selectStudent(s){

    selectedId = s.student_id;


    if (usingAPI){
        try{
            const res = await fetchWithTimeout(`${API_URL}/students/${s.student_id}`);
            const fresh = await res.json();
            if(fresh && !fresh.error) s = fresh;

        }catch (err){
            console.warn('Could not fetch fresh student data, using cached');
        }
    }

}

//Update sidebar selection highlights
document.querySelectorAll('.student-row').forEach(r => {
    r.classList.toggle('selected',r.dataset.id === selectedId);
});

//Show detail panel

document.getElementById('placeholder').classList.add('hidden');
document.getElementById('detailContent').classList.remove('hidden');

//Student header
document.getElementById('dStudentId').textContent = 
 s.student_id.replace('OULAD_','OULAD /');

document.getElementById('dRiskHeadline').textContent = 
`Week 17 Academic Risk: ${(s.academic_risk * 100).toFixed(1)}%`; 


const badge = document.getElementById('dAlertBadge');
badge.textContent = s.alert_level;
badge.className = 'alert-badge-large';
const badgeColor = {
    HIGH:   { bg: '#FCEBEB', color: '#C00000', border: '#FECACA' },
    MEDIUM: { bg: '#FFF3E0', color: '#B86B00', border: '#FDE68A' },
    LOW:    { bg: '#E8F5E9', color: '#2E7D32', border: '#A7F3D0' }
};

const bc = badgeColor[s.alert_level] || badgeColor.LOW;
badge.style.backgroundColor = bc.bg;
badge.style.color = bc.color;
badge.style.border = `1px solid ${bc.border}`;

// 4 Horizon cards

[
    { id: 'dW4',  bar: 'bW4',  val: s.week4_risk  },
    { id: 'dW8',  bar: 'bW8',  val: s.week8_risk  },
    { id: 'dW12', bar: 'bW12', val: s.week12_risk },
    { id: 'dW17', bar: 'bW17', val: s.week17_risk }
  ].forEach(h => {
    const pct   = (h.val * 100).toFixed(1);
    const color = riskColor(h.val);
    document.getElementById(h.id).textContent = pct + '%';
    document.getElementById(h.id).style.color = color;
    const bar = document.getElementById(h.bar);
    bar.style.width      = pct + '%';
    bar.style.background = color;
  });

//XAI Reason
document.getElementById('dReason').textContent = 'Model flags: ' + s.main_reason;

//Feature bars

const featureBars = document.getElementById('featureBars');
  featureBars.innerHTML = '';
  [
    { key: s.top_feature_1, pct: s.top_feature_1_pct, rank: 'rank1' },
    { key: s.top_feature_2, pct: s.top_feature_2_pct, rank: 'rank2' },
    { key: s.top_feature_3, pct: s.top_feature_3_pct, rank: 'rank3' }
  ].forEach(f => {
    const row       = document.createElement('div');
    row.className   = 'feature-bar-row';
    const width     = ((f.pct / 35) * 100).toFixed(1);
    row.innerHTML = `
      <span class="feature-bar-label">${FEATURE_NAMES[f.key] || f.key}</span>
      <div class="feature-bar-track">
        <div class="feature-bar-fill ${f.rank}" style="width:${width}%"></div>
      </div>
      <span class="feature-bar-pct">${f.pct}%</span>
    `;
    featureBars.appendChild(row);
  });










