"""
Stock Sentiment Analysis Module

This module orchestrates the process of fetching news articles about specified companies
and analyzing their sentiment to gauge market perception. It uses specialized agents
for news retrieval and sentiment analysis to provide insights about company sentiment
in news media.

The module serves as the main entry point for the stock sentiment analysis application.
"""

from agents.news_agent import NewsAgent
from agents.sentiment_agent import SentimentAgent

def main():
    # Initialize agents
    news_agent = NewsAgent()
    sentiment_agent = SentimentAgent()

    # Specify the company for which to fetch news
    company_name = "Example Company"

    # Fetch news articles
    articles = news_agent.fetch_news(company_name)

    # Analyze sentiment of the fetched articles
    for article in articles:
        sentiment_score, analysis = sentiment_agent.analyze_sentiment(article['content'])
        print(f"Title: {article['title']}")
        print(f"Sentiment Score: {sentiment_score}")
        print(f"Analysis: {analysis}\n")

if __name__ == "__main__":
    main()