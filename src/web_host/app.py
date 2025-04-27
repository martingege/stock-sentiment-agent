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
import csv
from datetime import datetime
import json
import re


# Environment variables
NEWS_API_KEY = os.getenv("NEWS_API_KEY")  # store this in your .env
NEWS_API_URL = "https://newsapi.org/v2/everything"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
app = Flask(__name__)

# Configurations Variables
use_ai = True  # Set to False to disable AI features
model = "gpt-4o"  # Use the latest model for better performance

# Prompt Config
# Common HTML formatting instruction for LLM prompts
common_html_instruction = (
    "Use only basic HTML tags like <h3>, <p>, <ul>, and <li>.\n"
    "Structure the response into clear sections as requested.\n"
    "Do NOT include any <html>, <head>, or <body> tags. Keep it clean for embedding.\n\n"
)

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
        f"{common_html_instruction}"
        f"Structure your output like this:\n"
        f"1. <h3>Overview</h3> – Introduce the company briefly.\n"
        f"2. <h3>Current Technical Overview</h3> – Explain price vs moving average, RSI, and volume trends.\n"
        f"3. <h3>Recommendation and Confidence</h3> – Explain the recommendation (BUY/SELL/HOLD) and what the confidence score implies.\n\n"
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
        content = response.choices[0].message.content.strip()
        data_blob = {
            "ticker": ticker,
            "current_price": current_price,
            "moving_avg": moving_avg,
            "rsi": rsi,
            "latest_volume": latest_volume,
            "avg_volume": avg_volume,
            "recommendation": recommendation,
            "confidence_score": confidence_score
        }
        log_llm_usage(
            use_case="indicator",
            data_blob=data_blob,
            prompt_template=prompt,
            model_used=model,
            llm_reply=content
        )
        return content
    except Exception as e:
        return f"⚠️ Error generating explanation: {str(e)}"

# Section 2: News Headlines and Sentiment Analysis
def fetch_news_headlines(ticker):
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
        "of the following news headlines about a stock and provide a structured explanation.\n\n"
        "News Headlines:\n" +
        "\n".join(f"- {h}" for h in headlines) + 
        # f"{common_html_instruction}"
        "Structure the output as:\n"
        "1. <h3>Overview</h3> – Summarize the overall news tone and themes.\n"
        "2. <h3>Sentiment Summary</h3> – Was the sentiment mostly positive, negative, or neutral?\n"
        "3. <h3>Recommendation</h3> – Based on the news, should the user consider BUY, SELL, or HOLD? "
        "Include a confidence score (high/medium/low) and a brief reasoning.\n\n"
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a financial assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        content = response.choices[0].message.content.strip()
        log_llm_usage(use_case="news", data_blob="\n".join(headlines), prompt_template=prompt, model_used="gpt-4", llm_reply=content)
        return content
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
            "surprise": round(surprise, 4)
        }

    except Exception as e:
        print("Finnhub earnings error:", e)
        return {}

def generate_earnings_digest(ticker, earnings_info):
    if not earnings_info or not earnings_info.get("latest_date"):
        return "No recent earnings report available."

    prompt = (
        f"Generate an easy-to-understand explanation for a stock's earnings report for ticker {ticker}.\n\n"
        f"Use the following data:\n"
        f"- Latest earnings date: {earnings_info['latest_date']}\n"
        f"- Expected EPS: {earnings_info['expected_eps']}\n"
        f"- Actual EPS: {earnings_info['actual_eps']}\n"
        f"- Surprise: {earnings_info['surprise']:.2f}\n"
        f"- Next earnings date: {earnings_info['next_earnings']}\n\n"
        f"{common_html_instruction}"
        f"Structure the response into these sections:\n"
        f"1. <h3>Summary</h3> – What happened in this earnings report.\n"
        f"2. <h3>Impact on Stock</h3> – How such earnings surprises usually affect the stock.\n"
        f"3. <h3>Recommendation</h3> – Should the user consider Buy, Hold, or Sell based on this report (explain simply)?\n\n"
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You're a financial analyst who explains things clearly for average people."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.6,
            max_tokens=2000
        )

        content = response.choices[0].message.content.strip()
        log_llm_usage(use_case="earnings", data_blob=earnings_info, prompt_template=prompt, model_used="gpt-4o", llm_reply=content)
        return content
    except Exception as e:
        return f"⚠️ Error generating earnings digest: {e}"


def final_stock_action(ticker, technical_explanation, news_sentiment_explanation, earnings_digest_explanation):
    prompt = f"""
        You are a financial assistant. Your job is to make a final stock recommendation based on the following inputs:

        1. Technical Indicators Analysis:
        {technical_explanation}

        2. News Sentiment Analysis:
        {news_sentiment_explanation}

        3. Earnings Digest Analysis:
        {earnings_digest_explanation}

        Instructions:
        - Analyze all 3 inputs carefully.
        - Make a final recommendation: 'Buy', 'Hold', or 'Sell'.
        - Assign a confidence score between 0% and 100% based on how strong the combined evidence is.
        - Write a short explanation in easy, layman's terms why you made that decision.
        - Only return a valid JSON in this exact format. No other text. like this:
        {{
        "final_decision": "Buy",
        "confidence_score": 85,
        "explanation": "Explain the rationale here..."
        }}

        Be objective, and act as a rational financial advisor helping a beginner investor.
        """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful financial analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1200
        )
        reply_text = response.choices[0].message.content.strip()
        # 🛡️ Sanitize reply if wrapped with ```json``` ... ```
        if reply_text.startswith("```"):
            reply_text = re.sub(r"```(?:json)?", "", reply_text)  # Remove opening ```
            reply_text = reply_text.replace("```", "")             # Remove closing ```
            reply_text = reply_text.strip()

        result = json.loads(reply_text)

        data_blob = {
            "technical_explanation": technical_explanation, 
            "news_sentiment_explanation": news_sentiment_explanation, 
            "earnings_digest_explanation": earnings_digest_explanation
        }
        log_llm_usage(use_case="final", data_blob=data_blob, prompt_template=prompt, model_used=model, llm_reply=reply_text)
        return result["final_decision"], result["confidence_score"], result["explanation"]

    except Exception as e:
        print(f"⚠️ Error in final_stock_action LLM call: {e}")
        return "Hold", 50, "There was an error processing the final decision."


# LLM Usage Logging
def log_llm_usage(use_case, data_blob, prompt_template, model_used, llm_reply):
    """
    Logs LLM usage details into a CSV file for tracking and analysis.
    """
    log_file_path = os.path.join(os.path.dirname(__file__), "../../llm_usage_log.csv")
    log_file_path = os.path.abspath(log_file_path)

    headers = ["timestamp", "use_case", "data_blob", "prompt_template", "model_used", "llm_reply"]
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    # Check if file exists to determine if headers are needed
    file_exists = os.path.isfile(log_file_path)

    with open(log_file_path, mode="a", newline='', encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": timestamp,
            "use_case": use_case,
            "data_blob": data_blob,
            "prompt_template": prompt_template,
            "model_used": model_used,
            "llm_reply": llm_reply
        })

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
    final_decision = final_confidence_score = final_decision_explanation = None

    if request.method == 'POST':
        ticker = request.form['ticker'].upper()
        indicators = gather_indicators(ticker)

        if indicators:
            current_price = round(indicators['current_price'], 4)
            moving_avg = round(indicators['moving_avg'], 4)
            rsi = round(indicators['rsi'], 4)
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
        if use_ai and earnings_info:
            earnings_digest = generate_earnings_digest(ticker, earnings_info)

        # Generate final recommendation
        final_decision, final_confidence_score, final_decision_explanation = final_stock_action(ticker, explanation, news_sentiment, earnings_digest)
        robinhood_url = f"https://robinhood.com/stocks/{ticker.upper()}"


    return render_template(
        'index_modern.html',
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
        short_term_signals=short_term_signals,
        final_decision = final_decision,
        final_confidence_score = final_confidence_score,
        final_decision_explanation = final_decision_explanation,
        robinhood_url = robinhood_url
    )

if __name__ == '__main__':
    app.run(debug=True)