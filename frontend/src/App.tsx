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
  ticker?: string
  closes_used?: number[]
}

const formatProbability = (value: number) => `${(value * 100).toFixed(1)}%`

function App() {
  const apiBase = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000'

  const [tickerInput, setTickerInput] = useState<string>('AAPL')
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

  const requestPrediction = async (endpoint: string, payload: Record<string, unknown>) => {
    setPredictError(null)
    setIsPredicting(true)
    try {
      const response = await fetch(`${apiBase}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
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

  const handlePredictFromTicker = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()

    const ticker = tickerInput.trim()
    if (!ticker) {
      setPredictError('Please enter a ticker symbol (e.g. AAPL).')
      return
    }

    await requestPrediction('/predict-direction-from-ticker', { ticker })
  }

  return (
    <div className="page">
      <header className="hero">
        <p className="eyebrow">LightGBM direction predictor</p>
        <h1>Directional Forecasting using Tree Model</h1>
        <p className="lede">
          We pull the latest daily closes with yfinance, build return/volatility features, and score the next-day move
          for you.
        </p>
      </header>

      <div className="grid">
        <form className="panel" onSubmit={handlePredictFromTicker}>
          <div className="panel-head">
            <div>
              <p className="eyebrow subtle">Input</p>
              <h2>Ticker lookup</h2>
            </div>
            <span className="badge">
              {isLoadingInfo ? 'Loading model…' : info ? `Trained on ${info.trained_on_ticker}` : 'Model info unavailable'}
            </span>
          </div>

          <label className="label" htmlFor="ticker">
            Ticker (we&apos;ll fetch latest closes)
          </label>
          <div className="ticker-row">
            <div className="ticker-inputs">
              <input
                id="ticker"
                value={tickerInput}
                onChange={(event) => setTickerInput(event.target.value)}
                placeholder="MSFT"
                className="text-input"
              />
              <p className="help">Uses the most recent daily closes (default 30-day window) via yfinance.</p>
            </div>
          </div>

          {predictError && <div className="alert">{predictError}</div>}

          <div className="actions">
            <button type="submit" disabled={isPredicting}>
              {isPredicting ? 'Scoring…' : 'Predict direction'}
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
              {prediction.ticker && (
                <p className="muted">
                  Source: {prediction.ticker} ({prediction.closes_used ? `${prediction.closes_used.length} closes` : 'latest closes'})
                </p>
              )}
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
