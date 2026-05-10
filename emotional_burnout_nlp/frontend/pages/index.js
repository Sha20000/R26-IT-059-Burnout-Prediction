import { useState, useEffect, useCallback } from 'react'
import Head from 'next/head'
import Header from '../src/components/Header'
import StudentList from '../src/components/StudentList'
import StudentDetail from '../src/components/StudentDetail'
import { STUDENTS } from '../src/data/students'

export default function Dashboard() {
  const [selected, setSelected] = useState(null)
  const [results, setResults] = useState({})

  const analyze = useCallback(async (student) => {
    if (results[student.id] && results[student.id] !== 'error') return

    setResults(prev => ({ ...prev, [student.id]: 'loading' }))
    try {
      const res = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: student.text }),
      })
      const data = await res.json()
      setResults(prev => ({ ...prev, [student.id]: data }))
    } catch {
      setResults(prev => ({ ...prev, [student.id]: 'error' }))
    }
  }, [results])

  const handleSelect = useCallback((student) => {
    setSelected(student)
    analyze(student)
  }, [analyze])

  useEffect(() => {
    STUDENTS.forEach((s, i) => {
      setTimeout(() => analyze(s), i * 600)
    })
  }, [])

  const selectedResult = selected ? results[selected.id] : null

  return (
    <>
      <Head>
        <title>Emotional Burnout Detection — R26-IT-059</title>
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </Head>

      <div style={{
        display: 'flex',
        flexDirection: 'column',
        height: '100vh',
        background: '#0f172a',
        color: '#f1f5f9',
        fontFamily: "'Inter', 'Segoe UI', system-ui, sans-serif",
        overflow: 'hidden',
      }}>
        <Header />

        <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
          <StudentList
            students={STUDENTS}
            selected={selected}
            onSelect={handleSelect}
            results={results}
          />
          <div style={{ flex: 1, overflowY: 'auto', background: '#0f172a' }}>
            <StudentDetail student={selected} result={selectedResult} />
          </div>
        </div>
      </div>

      <style>{`
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
      `}</style>
    </>
  )
}
