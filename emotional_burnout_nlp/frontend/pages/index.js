import { useState, useEffect } from 'react'
import Head from 'next/head'
import Header from '../src/components/Header'
import StudentList from '../src/components/StudentList'
import StudentDetail from '../src/components/StudentDetail'
import { loadStudents, saveStudents } from '../src/data/students'

export default function Dashboard() {
  const [students, setStudents]   = useState([])
  const [selected, setSelected]   = useState(null)
  const [detailTab, setDetailTab] = useState('analysis')  // 'analysis' | 'progression'

  useEffect(() => {
    setStudents(loadStudents())
  }, [])

  const handleSelect = (student) => {
    setSelected(student)
    setDetailTab('analysis')
  }

  const handleAddNew = () => {
    setSelected(null)
    setDetailTab('progression')
  }

  const handleStudentSaved = (newStudent) => {
    setStudents(prev => {
      const exists = prev.findIndex(s => s.id === newStudent.id)
      const updated = exists >= 0
        ? prev.map(s => s.id === newStudent.id ? newStudent : s)
        : [...prev, newStudent]
      saveStudents(updated)
      return updated
    })
    setSelected(newStudent)
    setDetailTab('analysis')
  }

  const handleClearAll = () => {
    if (!window.confirm('Remove all students? This cannot be undone.')) return
    saveStudents([])
    setStudents([])
    setSelected(null)
  }

  const selectedStudent = selected
    ? students.find(s => s.id === selected.id) || selected
    : null

  return (
    <>
      <Head>
        <title>Emotional Burnout Detection — R26-IT-059</title>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </Head>

      <div style={{
        display: 'flex', flexDirection: 'column', height: '100vh',
        background: '#f8fafc', color: '#0f172a',
        fontFamily: "'Inter','Segoe UI',system-ui,sans-serif",
        overflow: 'hidden',
      }}>
        <Header />

        <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
          <StudentList
            students={students}
            selected={selectedStudent}
            onSelect={handleSelect}
            onAddNew={handleAddNew}
            onClearAll={handleClearAll}
          />
          <div style={{ flex: 1, overflowY: 'auto', background: '#f8fafc' }}>
            <StudentDetail
              student={selectedStudent}
              addingNew={!selected}
              forcedTab={detailTab}
              onTabChange={setDetailTab}
              onStudentSaved={handleStudentSaved}
            />
          </div>
        </div>
      </div>

      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </>
  )
}
