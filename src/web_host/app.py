from flask import Flask, render_template, request
from indicators import gather_indicators, suggest_action_with_confidence
import yfinance as yf
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for matplotlib
import matplotlib.pyplot as plt
import io
import base64
import markdown
import os
import pandas_ta as ta
from openai import OpenAI
import requests

# Environment variables
NEWS_API_KEY = os.getenv("NEWS_API_KEY")  # store this in your .env
NEWS_API_URL = "https://newsapi.org/v2/everything"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
app = Flask(__name__)

# Configurations Variables
use_ai = True  # Set to False to disable AI features
model = "gpt-4o-mini"  # Use the latest model for better performance

# Seciont 1: Short Term Indicators and Recommendations
def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="3mo")
    return hist['Close']

def generate_chart(data, ticker):
    fig, ax = plt.subplots()
    data.plot(ax=ax, label='Close Price', linewidth=2)
    data.rolling(window=20).mean().plot(ax=ax, label='20-day MA')
    plt.title(f'{ticker} - Price and 20-Day Moving Average')
    plt.legend()

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def gpt_explanation(ticker, current_price, moving_avg, rsi, latest_volume, avg_volume, recommendation, confidence_score):
    """
    Generate a GPT-based stock explanation that considers technical indicators and recommendation.
    """
    prompt = (
        f"Generate a professional but beginner-friendly investment analysis for the stock {ticker}. "
        f"Here is the data you should use:\n\n"
        f"- Current Price: ${current_price:.2f}\n"
        f"- 20-Day Moving Average: ${moving_avg:.2f}\n"
        f"- RSI (Relative Strength Index): {rsi:.2f}\n"
        f"- Latest Volume: {latest_volume:.0f}\n"
        f"- 10-Day Average Volume: {avg_volume:.0f}\n"
        f"- Recommendation based on short-term technical indicators: {recommendation}\n"
        f"- Confidence Score: {confidence_score}%\n\n"
        f"Organize your explanation clearly using basic HTML tags only (<h2>, <p>, <ul>, <li>). "
        f"Structure your output like this:\n"
        f"1. <h2>Overview</h2> – Introduce the company briefly.\n"
        f"2. <h2>Current Technical Overview</h2> – Explain price vs moving average, RSI, and volume trends.\n"
        f"3. <h2>Recommendation and Confidence</h2> – Explain the recommendation (BUY/SELL/HOLD) and what the confidence score implies.\n\n"
        f"Do NOT include any <html>, <head>, or <body> tags. Keep it clean for embedding."
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a financial assistant who explains clearly and simply."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1200
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Error generating explanation: {str(e)}"

# Section 2: News Headlines and Sentiment Analysis
def fetch_news_headlines(ticker):
    print("API Key:", NEWS_API_KEY)  # ✅ Debug line
    params = {
        "q": ticker,
        "sortBy": "publishedAt",
        "language": "en",
        "pageSize": 5,
        "apiKey": NEWS_API_KEY
    }
    try:
        response = requests.get(NEWS_API_URL, params=params)
        articles = response.json().get("articles", [])
        headlines = [article["title"] for article in articles if "title" in article]
        return headlines
    except Exception as e:
        return [f"⚠️ Failed to fetch news: {e}"]
    
def analyze_news_sentiment(headlines):
    if not headlines:
        return "No headlines available for sentiment analysis."

    prompt = (
        "You are a financial assistant. Analyze the overall sentiment (positive, negative, or neutral) "
        "of the following news headlines about a stock and provide a one-paragraph summary:\n\n"
        + "\n".join(f"- {h}" for h in headlines)
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a financial assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Error generating sentiment: {str(e)}"

# Section 3: Long-Term Signals and Earnings Digest
def get_earnings_info(ticker):
    try:
        # 1. Get upcoming earnings date
        upcoming_url = f"https://finnhub.io/api/v1/calendar/earnings?symbol={ticker}&token={FINNHUB_API_KEY}"
        upcoming_response = requests.get(upcoming_url).json()
        next_earnings = None
        if upcoming_response.get("earningsCalendar"):
            next_earnings = upcoming_response["earningsCalendar"][0].get("date")

        # 2. Get recent earnings (EPS actual vs. estimate)
        earnings_url = f"https://finnhub.io/api/v1/stock/earnings?symbol={ticker}&token={FINNHUB_API_KEY}"
        earnings_response = requests.get(earnings_url).json()
        latest = earnings_response[0] if earnings_response else None

        latest_date = actual_eps = expected_eps = surprise = None
        if latest:
            latest_date = latest.get("period")
            actual_eps = latest.get("actual")
            expected_eps = latest.get("estimate")
            surprise = actual_eps - expected_eps if actual_eps and expected_eps else None

        return {
            "next_earnings": next_earnings,
            "latest_date": latest_date,
            "actual_eps": actual_eps,
            "expected_eps": expected_eps,
            "surprise": surprise
        }

    except Exception as e:
        print("Finnhub earnings error:", e)
        return {}

def generate_earnings_digest(ticker, earnings_info):
    if not earnings_info or not earnings_info.get("latest_date"):
        return "No recent earnings report available."

    prompt = (
        f"Generate an easy-to-understand HTML explanation for a stock's earnings report for ticker {ticker}.\n\n"
        f"Use the following data:\n"
        f"- Latest earnings date: {earnings_info['latest_date']}\n"
        f"- Expected EPS: {earnings_info['expected_eps']}\n"
        f"- Actual EPS: {earnings_info['actual_eps']}\n"
        f"- Surprise: {earnings_info['surprise']:.2f}\n"
        f"- Next earnings date: {earnings_info['next_earnings']}\n\n"
        f"Organize the explanation using simple HTML tags only (<h2>, <p>, <ul>, <li>). "
        f"Structure the response into these sections:\n"
        f"1. <h2>Summary</h2> – What happened in this earnings report.\n"
        f"2. <h2>Impact on Stock</h2> – How such earnings surprises usually affect the stock.\n"
        f"3. <h2>Recommendation</h2> – Should the user consider Buy, Hold, or Sell based on this report (explain simply)?\n\n"
        f"Do not include any <html>, <head>, or <body> tags."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You're a financial analyst who explains things clearly for average people."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.6,
            max_tokens=2000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Error generating earnings digest: {e}"


# Flask app setup
@app.route('/', methods=['GET', 'POST'])
def index():
    recommendation = None
    explanation = None
    current_price = None
    moving_avg = None
    chart = None
    ticker = ""
    news_headlines = None
    news_sentiment = None
    earnings_digest = None
    earnings_info = None
    short_term_signals = None
    rsi = None
    volume_spike = None
    latest_volume = None
    avg_volume = None
    confidence_score = None

    if request.method == 'POST':
        ticker = request.form['ticker'].upper()
        indicators = gather_indicators(ticker)

        if indicators:
            current_price = indicators['current_price']
            moving_avg = indicators['moving_avg']
            rsi = indicators['rsi']
            volume_spike = indicators['volume_spike']
            latest_volume = indicators['latest_volume']
            avg_volume = indicators['avg_volume']
            
            # Then continue your logic
            # like calculating recommendations, generating charts, GPT explanations, etc.
        else:
            # Handle the case where ticker data couldn't be fetched
            current_price = moving_avg = rsi = volume_spike = latest_volume = avg_volume = None
        
        recommendation, confidence_score = suggest_action_with_confidence(current_price, moving_avg, rsi, latest_volume, avg_volume)

        #  use AI features only if enabled
        if use_ai:
            explanation = gpt_explanation(ticker, current_price, moving_avg, rsi, latest_volume, avg_volume, recommendation, confidence_score)

        # Fetch news headlines and analyze sentiment
        news_headlines = fetch_news_headlines(ticker)
        if use_ai and news_headlines:
            news_sentiment = analyze_news_sentiment(news_headlines)
        
        # Generate long-term signals
        earnings_info = get_earnings_info(ticker)
        print("Earnings Info:", earnings_info)  # Debug line
        if use_ai and earnings_info:
            earnings_digest = generate_earnings_digest(ticker, earnings_info)


    return render_template(
        'index_mixed.html',
        ticker=ticker,
        recommendation=recommendation,
        current_price=current_price,
        moving_avg=moving_avg,
        rsi=rsi,
        volume_spike=volume_spike,
        latest_volume=latest_volume,
        avg_volume=avg_volume,
        confidence_score=confidence_score,
        explanation=explanation,
        chart=chart,
        news_headlines=news_headlines,
        news_sentiment=news_sentiment,
        earnings_info=earnings_info,
        earnings_digest=earnings_digest,
        short_term_signals=short_term_signals
    )

if __name__ == '__main__':
    app.run(debug=True)