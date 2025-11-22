import { FormEvent, useEffect, useState } from 'react'
import './App.css'
import './Output.css'

type PredictorInfo = {
  trained_on_ticker: string
  features_expected: string[]
}

type Prediction = {
  direction: 'up' | 'down'
  prob_up: number
  prob_down: number
}

const DEFAULT_CLOSES =
  '421.1, 422.4, 421.8, 423.6, 424.9, 426.2, 427.1, 428.5, 429.2, 430.4, 431.0'

const parseCloses = (raw: string): number[] =>
  raw
    .split(/[\s,]+/)
    .map((value) => Number.parseFloat(value))
    .filter((value) => Number.isFinite(value))

const formatProbability = (value: number) => `${(value * 100).toFixed(1)}%`

function App() {
  const apiBase = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000'

  const [closesInput, setClosesInput] = useState<string>(DEFAULT_CLOSES)
  const [info, setInfo] = useState<PredictorInfo | null>(null)
  const [infoError, setInfoError] = useState<string | null>(null)
  const [prediction, setPrediction] = useState<Prediction | null>(null)
  const [predictError, setPredictError] = useState<string | null>(null)
  const [isPredicting, setIsPredicting] = useState<boolean>(false)
  const [isLoadingInfo, setIsLoadingInfo] = useState<boolean>(false)

  const fetchPredictorInfo = async () => {
    setIsLoadingInfo(true)
    setInfoError(null)
    try {
      const response = await fetch(`${apiBase}/predict-info`)
      if (!response.ok) {
        throw new Error(`Backend responded with ${response.status}`)
      }

      const data: PredictorInfo = await response.json()
      setInfo(data)
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Could not load predictor info'
      setInfoError(message)
    } finally {
      setIsLoadingInfo(false)
    }
  }

  useEffect(() => {
    fetchPredictorInfo()
  }, [])

  const handlePredict = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    setPredictError(null)

    const closes = parseCloses(closesInput)
    if (closes.length < 11) {
      setPredictError('Please provide at least 11 closing prices (oldest to newest).')
      return
    }

    setIsPredicting(true)
    try {
      const response = await fetch(`${apiBase}/predict-direction`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ closes }),
      })

      if (!response.ok) {
        const detail = await response.json().catch(() => ({}))
        const message = typeof detail.detail === 'string' ? detail.detail : `Backend responded with ${response.status}`
        throw new Error(message)
      }

      const data: Prediction = await response.json()
      setPrediction(data)
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Prediction request failed'
      setPredictError(message)
    } finally {
      setIsPredicting(false)
    }
  }

  return (
    <div className="page">
      <header className="hero">
        <p className="eyebrow">LightGBM direction predictor</p>
        <h1>Send a price window and get the model&apos;s bias</h1>
        <p className="lede">
          Paste at least 11 recent closing prices (oldest first). The backend builds returns/volatility features and
          scores the next-day move for you.
        </p>
      </header>

      <div className="grid">
        <form className="panel" onSubmit={handlePredict}>
          <div className="panel-head">
            <div>
              <p className="eyebrow subtle">Input</p>
              <h2>Price window</h2>
            </div>
            <span className="badge">
              {isLoadingInfo ? 'Loading model…' : info ? `Trained on ${info.trained_on_ticker}` : 'Model info unavailable'}
            </span>
          </div>

          <label className="label" htmlFor="closes">
            Recent closes
          </label>
          <textarea
            id="closes"
            value={closesInput}
            onChange={(event) => setClosesInput(event.target.value)}
            placeholder="112.5, 112.9, 113.4, 114.1, 113.8, 114.6, 115.2, 115.5, 116.1, 116.9, 117.4"
          />
          <p className="help">Comma, space, or newline separated. Minimum of 11 values.</p>

          {predictError && <div className="alert">{predictError}</div>}

          <div className="actions">
            <button type="submit" disabled={isPredicting}>
              {isPredicting ? 'Scoring…' : 'Predict direction'}
            </button>
            <button type="button" className="ghost" onClick={() => setClosesInput(DEFAULT_CLOSES)}>
              Use sample window
            </button>
          </div>
        </form>

        <section className="panel">
          <div className="panel-head">
            <div>
              <p className="eyebrow subtle">Result</p>
              <h2>Prediction</h2>
            </div>
            {prediction && <span className={`chip ${prediction.direction}`}>{prediction.direction} bias</span>}
          </div>

          {isPredicting && <p className="muted">Scoring your window against the model…</p>}

          {!isPredicting && prediction && (
            <div className="result">
              <p className="direction">
                Likely <strong>{prediction.direction.toUpperCase()}</strong>
              </p>
              <div className="prob-row">
                <span className="muted">Up</span>
                <div className="bar">
                  <div className="fill up" style={{ width: `${Math.round(prediction.prob_up * 100)}%` }} />
                </div>
                <span className="value">{formatProbability(prediction.prob_up)}</span>
              </div>
              <div className="prob-row">
                <span className="muted">Down</span>
                <div className="bar">
                  <div className="fill down" style={{ width: `${Math.round(prediction.prob_down * 100)}%` }} />
                </div>
                <span className="value">{formatProbability(prediction.prob_down)}</span>
              </div>
            </div>
          )}

          {!isPredicting && !prediction && <p className="muted">Send some closes to see the model&apos;s probabilities.</p>}

          <div className="info">
            <h3>Model inputs</h3>
            {infoError && <div className="alert">{infoError}</div>}
            {isLoadingInfo && <p className="muted">Loading predictor metadata…</p>}
            {info && (
              <ul className="chips">
                {info.features_expected.map((feature) => (
                  <li key={feature} className="chip neutral">
                    {feature}
                  </li>
                ))}
              </ul>
            )}
          </div>
        </section>
      </div>
    </div>
  )
}

export default App
