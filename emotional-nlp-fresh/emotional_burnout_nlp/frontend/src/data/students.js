// Students are stored in localStorage under key "burnout_students"
// Each student: { id, name, weeks: { 2, 4, 8, 18 }, latestResult, worstResult }

export function loadStudents() {
  if (typeof window === 'undefined') return []
  try {
    return JSON.parse(localStorage.getItem('burnout_students') || '[]')
  } catch {
    return []
  }
}

export function saveStudents(students) {
  if (typeof window === 'undefined') return
  localStorage.setItem('burnout_students', JSON.stringify(students))
}
