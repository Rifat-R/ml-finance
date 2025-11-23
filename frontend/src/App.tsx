import type { FormEvent } from 'react'
import { useEffect, useState } from 'react'

type PredictorInfo = {
  features_expected: string[]
  cached_models?: string[]
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
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <div className="mx-auto flex max-w-6xl flex-col gap-6 px-4 py-10">
        <header className="space-y-2">
          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-cyan-300">LightGBM direction predictor</p>
          <h1 className="text-3xl font-bold leading-tight md:text-4xl">Directional Forecasting using Tree Model</h1>
          <p className="max-w-3xl text-slate-400">
            We pull the latest daily closes with yfinance, build return/volatility features, and score the next-day move
            for you.
          </p>
          {prediction && (
            <p className="text-sm text-slate-400">
              Trained on demand for <span className="font-semibold text-cyan-300">{prediction.ticker}</span>; cached for
              future calls.
            </p>
          )}
        </header>

        <div className="grid gap-4 md:grid-cols-2 md:gap-6">
          <form
            className="rounded-2xl border border-slate-800 bg-slate-900/60 p-5 shadow-lg shadow-cyan-500/5 backdrop-blur"
            onSubmit={handlePredictFromTicker}
          >
            <div className="mb-4 flex items-center justify-between gap-3">
              <div>
                <p className="text-xs uppercase tracking-[0.18em] text-slate-400">Input</p>
                <h2 className="text-xl font-semibold text-slate-50">Ticker lookup</h2>
              </div>
              <span className="inline-flex items-center rounded-full border border-cyan-300/60 bg-cyan-400/10 px-3 py-1 text-sm font-semibold text-cyan-200">
                {prediction?.ticker
                  ? `Model for ${prediction.ticker}`
                  : isLoadingInfo
                    ? 'Loading model…'
                    : info?.cached_models?.length
                      ? `Cached: ${info.cached_models.length}`
                      : 'No cached models'}
              </span>
            </div>

            <label className="mb-2 block text-sm font-semibold text-slate-200" htmlFor="ticker">
              Ticker (we&apos;ll fetch latest closes)
            </label>
            <div className="space-y-2">
              <input
                id="ticker"
                value={tickerInput}
                onChange={(event) => setTickerInput(event.target.value)}
                placeholder="MSFT"
                className="w-full rounded-lg border border-slate-800 bg-slate-900/80 px-3 py-3 text-base text-slate-100 shadow-inner shadow-black/20 outline-none ring-1 ring-transparent transition focus:border-cyan-400/70 focus:ring-cyan-400/40"
              />
              <p className="text-sm text-slate-400">Uses the most recent daily closes (default 30-day window) via yfinance.</p>
            </div>

            {predictError && (
              <div className="mt-4 rounded-lg border border-rose-400/60 bg-rose-500/10 px-3 py-2 text-sm text-rose-100">
                {predictError}
              </div>
            )}

            <div className="mt-5 flex flex-wrap items-center gap-3">
              <button
                type="submit"
                disabled={isPredicting}
                className="inline-flex cursor-pointer items-center justify-center rounded-lg bg-gradient-to-r from-cyan-400 to-sky-500 px-4 py-2.5 text-sm font-semibold text-slate-950 shadow-lg shadow-cyan-500/30 transition hover:translate-y-[-1px] hover:shadow-cyan-400/40 disabled:translate-y-0 disabled:opacity-70 disabled:shadow-none"
              >
                {isPredicting ? 'Scoring…' : 'Predict direction'}
              </button>
            </div>
          </form>

          <section className="rounded-2xl border border-slate-800 bg-slate-900/60 p-5 shadow-lg shadow-cyan-500/5 backdrop-blur">
            <div className="mb-4 flex items-center justify-between gap-3">
              <div>
                <p className="text-xs uppercase tracking-[0.18em] text-slate-400">Result</p>
                <h2 className="text-xl font-semibold text-slate-50">Prediction</h2>
              </div>
              {prediction && (
                <span
                  className={`inline-flex items-center rounded-full px-3 py-1 text-sm font-semibold ${prediction.direction === 'up'
                      ? 'border border-emerald-300/60 bg-emerald-400/15 text-emerald-200'
                      : 'border border-amber-300/60 bg-amber-400/15 text-amber-200'
                    }`}
                >
                  {prediction.direction} bias
                </span>
              )}
            </div>

            {isPredicting && <p className="text-sm text-slate-400">Scoring your window against the model…</p>}

            {!isPredicting && prediction && (
              <div className="space-y-3">
                <p className="text-lg font-semibold text-slate-50">
                  Likely <span className="tracking-wide text-slate-100">{prediction.direction.toUpperCase()}</span>
                </p>
                {prediction.ticker && (
                  <p className="text-sm text-slate-400">
                    Source: {prediction.ticker} ({prediction.closes_used ? `${prediction.closes_used.length} closes` : 'latest closes'})
                  </p>
                )}
                <div className="space-y-2">
                  <div className="flex items-center gap-3 text-sm text-slate-300">
                    <span className="w-12 text-slate-400">Up</span>
                    <div className="h-2 flex-1 overflow-hidden rounded-full bg-slate-800">
                      <div className="h-full rounded-full bg-gradient-to-r from-emerald-400 to-emerald-500" style={{ width: `${Math.round(prediction.prob_up * 100)}%` }} />
                    </div>
                    <span className="w-16 text-right font-semibold text-slate-100">{formatProbability(prediction.prob_up)}</span>
                  </div>
                  <div className="flex items-center gap-3 text-sm text-slate-300">
                    <span className="w-12 text-slate-400">Down</span>
                    <div className="h-2 flex-1 overflow-hidden rounded-full bg-slate-800">
                      <div className="h-full rounded-full bg-gradient-to-r from-amber-400 to-rose-500" style={{ width: `${Math.round(prediction.prob_down * 100)}%` }} />
                    </div>
                    <span className="w-16 text-right font-semibold text-slate-100">{formatProbability(prediction.prob_down)}</span>
                  </div>
                </div>
              </div>
            )}

            {!isPredicting && !prediction && (
              <p className="text-sm text-slate-400">Submit a ticker to see the model&apos;s probabilities.</p>
            )}

            <div className="mt-6 border-t border-slate-800 pt-4">
              <h3 className="text-base font-semibold text-slate-100">Model inputs</h3>
              {infoError && (
                <div className="mt-2 rounded-lg border border-rose-400/60 bg-rose-500/10 px-3 py-2 text-sm text-rose-100">
                  {infoError}
                </div>
              )}
              {isLoadingInfo && <p className="text-sm text-slate-400">Loading predictor metadata…</p>}
              {info && (
                <ul className="mt-3 flex flex-wrap gap-2">
                  {info.features_expected.map((feature) => (
                    <li key={feature} className="rounded-full border border-slate-800 bg-slate-800/60 px-3 py-1 text-sm font-semibold text-slate-200">
                      {feature}
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </section>
        </div>
      </div>
    </div>
  )
}

export default App
