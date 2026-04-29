# Machine Learning in Financial Mathematics

A dissertation project that builds a directional forecaster for daily equity prices using LightGBM. The backend trains and serves per-ticker classifiers on demand; the frontend lets you submit a ticker, choose whether to use news sentiment, and view the predicted next-day direction along with a walk-forward backtest.

## How it works

- **Backend (FastAPI)**: on the first request for a ticker, it downloads daily prices via Tiingo, builds price features (rolling returns, volatility, momentum, MA distance, RSI), optionally merges in FinBERT-scored news sentiment, trains a LightGBM classifier with walk-forward validation, caches the artifact under `models/`, and serves predictions. Subsequent requests reuse the cached model.
- **Frontend (Vite + React + Tailwind)**: single-page UI to enter a ticker and toggle sentiment features. Displays the predicted direction, class probabilities, model accuracy, and a walk-forward backtest curve (model vs buy & hold).
- **Two modes**:
  - *Sentiment on*: price + news sentiment features. Trains and backtests on roughly 2020 onwards (news data is limited to ~3 years).
  - *Sentiment off*: price-only features. Trains on 2015 onwards and backtests every full year from 2018 to last year.

## Requirements

- Python 3.13+ (managed by `uv`)
- Node.js 18+ and npm
- [`uv`](https://docs.astral.sh/uv/) for Python dependency management. Install it with:
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```

## Setup

1. **Install backend dependencies** (creates `.venv/` and installs everything from `pyproject.toml` / `uv.lock`):
   ```bash
   uv sync
   ```

2. **Install frontend dependencies**:
   ```bash
   cd frontend
   npm install
   cd ..
   ```

3. **Configure environment variables** in a `.env` file at the repo root:
   ```
   TIINGO_API_KEY=your_tiingo_key
   NEWS_API_KEY=your_newsapi_key
   HF_TOKEN=your_huggingface_token
   ```
   - `TIINGO_API_KEY` is required for fetching historical prices.
   - `NEWS_API_KEY` is required only when sentiment mode is enabled (live inference fetches recent headlines).
   - `HF_TOKEN` is used by `transformers` to download the FinBERT model.

## Running the app

### Option A: one command

```bash
./run.sh
```

This starts the FastAPI backend on port 8000 and the Vite dev server. When you stop the frontend (Ctrl-C), the backend is killed too. Open the URL Vite prints (typically <http://127.0.0.1:5173>).

### Option B: two terminals

Backend:
```bash
uv run uvicorn backend.main:app --reload --port 8000
```

Frontend:
```bash
cd frontend
npm run dev
```

## Project layout

```
backend/
  main.py              FastAPI app entrypoint
  features.py          Feature definitions (price + sentiment) and frame builders
  training.py          Walk-forward training, evaluation, and yearly backtest
  routers/
    predictor.py       /predict-direction-from-ticker, /backtest-walk-forward, /ticker-info
  data/                Tiingo client and data fetching helpers
frontend/              Vite + React + Tailwind UI
models/                Cached trained model artifacts (gitignored)
scripts/               One-off scripts for news data preprocessing
```

## Notes

- Trained models are cached per ticker and per mode, e.g. `models/lgbm_direction_AAPL.pkl` (sentiment) and `models/lgbm_direction_AAPL_price_only.pkl` (price-only).
- To retrain a ticker, delete its pickle from `models/` and submit a new prediction.
