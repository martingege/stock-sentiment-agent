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

app = Flask(__name__)


def get_stock_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="3mo")
    return hist['Close']


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

    return render_template('index_ck.html',
                           ticker=ticker,
                           recommendation=recommendation,
                           current_price=current_price,
                           moving_avg=moving_avg,
                           explanation=explanation,
                           chart=chart,
                           news_headlines=news_headlines,
                           news_sentiment=news_sentiment)


if __name__ == '__main__':
    app.run(debug=True)