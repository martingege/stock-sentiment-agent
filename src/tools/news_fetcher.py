def fetch_news(company_name: str):
    import requests

    # Example API endpoint (replace with a real news API)
    api_url = f"https://newsapi.org/v2/everything?q={company_name}&apiKey=YOUR_API_KEY"
    
    response = requests.get(api_url)
    
    if response.status_code == 200:
        articles = response.json().get('articles', [])
        return articles
    else:
        raise Exception("Failed to fetch news articles")