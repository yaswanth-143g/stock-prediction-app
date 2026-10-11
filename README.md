# NSE Stock Price Prediction & Quantitative Portfolio Platform

An enterprise-grade stock price prediction, quantitative technical analysis, and portfolio intelligence dashboard built for the National Stock Exchange (NSE) of India.

## Features
- **Multi-Horizon ML Prediction:** 7-day multi-output forecasting powered by Random Forest, XGBoost, and LightGBM models trained on 36 canonical technical indicators.
- **FinBERT NLP News Sentiment Engine:** Real-time financial headline sentiment scoring with dynamic impact weighting ($\alpha$).
- **Live Market Feeds & Intraday Signals:** Upstox API v2 & yfinance integration with real-time WebSocket streaming and 5-minute intraday target projections.
- **Interactive Technical Visualizations:** TradingView-style interactive candlestick charts, moving averages (EMA/SMA 20, 50, 200), Bollinger Bands, RSI, and MACD.
- **Quantitative Risk & Backtesting:** Monte Carlo simulation, value-at-risk, Sharpe ratios, and historical rule-based strategy backtesting.

## Tech Stack
- **Backend:** Python 3, Flask, Flask-SocketIO, Gunicorn
- **Machine Learning:** scikit-learn, XGBoost, LightGBM, Hugging Face Transformers (FinBERT)
- **Frontend:** Vanilla JavaScript, Modern CSS Glassmorphism, Plotly / Lightweight Charts
- **Database / Cache:** SQLite / MySQL 8.0, Redis / In-Memory TTL Cache

## Quickstart

### Local Setup
```bash
# Clone the repository
git clone https://github.com/yaswanth-143g/stock-prediction-app.git
cd stock-prediction-app

# Install dependencies
pip install -r requirements.txt

# Run the platform
python run_server.py
```
Open your browser at `http://localhost:5000`.

### Production Deployment
The repository includes production deployment configurations:
- `Procfile` for Render / Railway / Heroku
- `Dockerfile` & `docker-compose.yml` for containerized hosting
