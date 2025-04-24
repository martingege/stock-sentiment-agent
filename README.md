# 🧠 Stock Sentiment Agent

A personalized stock analysis web app that leverages GPT-4, real-time market data, news sentiment, and earnings reports to help investors make smarter decisions in plain English.

---

## 🔍 Features

- **📈 Stock Trend Insight**  
  Pulls 3-month historical price data and calculates a 20-day moving average to offer simple buy/sell/hold recommendations.

- **🤖 GPT-Powered Investment Explanations**  
  Uses OpenAI’s GPT-4 to explain technical trends in plain, beginner-friendly language.

- **📰 News Sentiment Analysis**  
  Fetches the latest headlines using NewsAPI and summarizes sentiment using GPT.

- **📊 Visual Price Charts**  
  Shows interactive stock charts with moving averages.

- **📆 Earnings Digest (via Finnhub.io)**  
  Provides layman-friendly summaries of quarterly earnings reports, highlights EPS surprises, and lists the next earnings date.

- **🌓 Light/Dark Theme Toggle**  
  Easily switch between a Credit Karma–style interface and a cyberpunk visual theme.

---

## 🧪 Tech Stack

- Python 3.11
- Flask (Web framework)
- yfinance (price data)
- Finnhub.io (earnings data) https://finnhub.io/docs/api
- matplotlib (chart rendering)
- OpenAI GPT-4 (financial explanations)
- NewsAPI (news headlines)
- HTML + Jinja2 + CSS (for UI and theming)

---

## 🚀 Run Locally

### 1. Clone the repo

```bash
git clone git@github.com:martingege/stock-sentiment-agent.git
cd stock-sentiment-agent

### 2. Install
pip install -r requirements.txt

### 3. Set your API keys
OPENAI_API_KEY=your_openai_key_here
NEWS_API_KEY=your_newsapi_key_here
FINNHUB_API_KEY=your_finnhub_key_here

### 4. Run the app
python src/web_host/app.py

Then open http://127.0.0.1:5000 in your browser.


