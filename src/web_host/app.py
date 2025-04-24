from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for matplotlib
import matplotlib.pyplot as plt
import io
import base64
import markdown
import os
from openai import OpenAI
import requests


NEWS_API_KEY = os.getenv("NEWS_API_KEY")  # store this in your .env
NEWS_API_URL = "https://newsapi.org/v2/everything"


client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

app = Flask(__name__)


def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="3mo")
    return hist['Close']

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

def calculate_moving_average(data, window=20):
    return data.rolling(window=window).mean()


def generate_recommendation(data):
    current_price = data.iloc[-1]
    moving_avg = calculate_moving_average(data).iloc[-1]
    if current_price > moving_avg:
        return "📈 Consider buying"
    elif current_price < moving_avg:
        return "📉 Consider selling"
    else:
        return "🟰 Hold"


def gpt_explanation(ticker, current_price, moving_avg):
    prompt = (
        f"The stock ticker is {ticker}. Its current price is ${current_price:.2f} "
        f"and its 20-day moving average is ${moving_avg:.2f}. "
        f"Generate a short investment analysis for a beginner investor and explain why they should "
        f"consider buying, selling, or holding."
    )

    try:
        response = client.chat.completions.create(model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a financial assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=200)
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Error generating explanation: {str(e)}"


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
            max_tokens=250
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Error generating sentiment: {str(e)}"

def generate_earnings_digest(ticker, earnings_info):
    if not earnings_info or not earnings_info.get("latest_date"):
        return "No recent earnings report available."

    prompt = (
        f"{ticker} recently reported earnings on {earnings_info['latest_date']}. "
        f"The expected EPS was {earnings_info['expected_eps']}, and the actual was {earnings_info['actual_eps']}, "
        f"with a surprise of {earnings_info['surprise']:.2f}. "
        f"Explain what this means for an average investor in simple terms. Also mention if the stock usually reacts to such surprises, "
        f"and give the date for the next earnings report: {earnings_info['next_earnings']}."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You're a financial analyst who explains things clearly for average people."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.6,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Error generating earnings digest: {e}"

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendation = None
    explanation = None
    current_price = None
    moving_avg = None
    chart = None
    ticker = ""

    if request.method == 'POST':
        ticker = request.form['ticker'].upper()
        data = get_stock_data(ticker)
        if not data.empty:
            current_price = data.iloc[-1]
            moving_avg = calculate_moving_average(data).iloc[-1]
            recommendation = generate_recommendation(data)
            raw_explanation = gpt_explanation(ticker, current_price, moving_avg)
            explanation = markdown.markdown(raw_explanation)
            chart = generate_chart(data, ticker)
        news_headlines = fetch_news_headlines(ticker)
        print("📰 Headlines:", news_headlines)  # ✅ Debug line
        news_sentiment = analyze_news_sentiment(news_headlines)
        earnings_info = get_earnings_info(ticker)
        earnings_digest = generate_earnings_digest(ticker, earnings_info)

    return render_template('index_mixed.html',
                           ticker=ticker,
                           recommendation=recommendation,
                           current_price=current_price,
                           moving_avg=moving_avg,
                           explanation=explanation,
                           chart=chart,
                           news_headlines=news_headlines,
                           news_sentiment=news_sentiment,
                           earnings_info=earnings_info,
                           earnings_digest=earnings_digest)


if __name__ == '__main__':
    app.run(debug=True)