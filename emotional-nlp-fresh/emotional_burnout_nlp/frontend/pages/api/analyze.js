import { spawn } from 'child_process'
import path from 'path'

export default function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*')
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS')
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type')

  if (req.method === 'OPTIONS') {
    return res.status(200).end()
  }

  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' })
  }

  const { text } = req.body
  if (!text || text.trim().split(/\s+/).length < 3) {
    return res.status(400).json({ error: 'Please enter at least a few words.' })
  }

  const scriptPath = path.join(process.cwd(), '..', 'predict.py')
  const pythonPath = process.platform === 'win32'
    ? 'C:\\Users\\ASUS\\anaconda3\\python.exe'
    : 'python'

  const py = spawn(pythonPath, [scriptPath, text.trim()], {
    cwd: path.join(process.cwd(), '..'),
  })

  let stdout = ''
  let stderr = ''

  py.stdout.on('data', (data) => { stdout += data.toString() })
  py.stderr.on('data', (data) => { stderr += data.toString() })

  py.on('close', (code) => {
    if (code !== 0) {
      console.error('predict.py stderr:', stderr)
      return res.status(500).json({ error: 'Model inference failed.', detail: stderr.slice(-2000) })
    }
    try {
      const result = JSON.parse(stdout.trim())
      if (result.error) return res.status(400).json(result)
      return res.status(200).json(result)
    } catch {
      return res.status(500).json({ error: 'Failed to parse model output.' })
    }
  })
}
