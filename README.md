# Machine Learning in Financial Mathematics

This repository is a small dissertation project that builds a **directional forecaster** for daily equity prices using tree-based models (LightGBM). The backend trains and serves per-ticker classifiers on demand; the frontend lets you submit a ticker, fetches recent prices, and returns the model’s up/down probabilities.

## How it works
- **Backend (FastAPI)**: on first request for a ticker, it downloads recent prices via `yfinance`, builds simple return/volatility features (1, 3, 5, 10-day means + 10-day volatility), trains a LightGBM classifier, caches it, and serves predictions. Subsequent requests reuse the cached/saved model for that ticker.
- **Frontend (Vite + React + Tailwind)**: single-page UI to enter a ticker and display the predicted direction and probabilities.

## Quickstart
1. **Create and activate a virtual environment** (Python 3.10+ recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate   # on Windows: venv\Scripts\activate
   ```
2. **Install backend dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Install frontend dependencies**:
   ```bash
   cd frontend
   npm install
   cd ..
   ```
4. **Run the app (backend + frontend)**:
   ```bash
   ./run.sh
   ```
   - The script starts FastAPI with uvicorn on port 8000 and then launches the Vite dev server.
   - Open the frontend URL printed by Vite (usually http://127.0.0.1:5173).
