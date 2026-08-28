// Proxies to the warm inference service (serve_model.py, port 5004), which
// keeps BERT loaded in memory. Previously this spawned a fresh Python process
// per request, which reloaded 418 MB of weights every time (~20-30s per call,
// and it ran the machine out of memory under any real load).
//
// Start the service first:
//   cd emotional_burnout_nlp && .venv/Scripts/python serve_model.py

const MODEL_API = process.env.MODEL_API_URL || 'http://localhost:5004'

export default async function handler(req, res) {
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

  try {
    const upstream = await fetch(`${MODEL_API}/api/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: text.trim() }),
    })

    const data = await upstream.json()
    return res.status(upstream.status).json(data)
  } catch (e) {
    console.error('Model service unreachable:', e.message)
    return res.status(503).json({
      error: `Model service not running (${MODEL_API}).`,
      detail: 'Start it with: .venv/Scripts/python serve_model.py',
    })
  }
}
