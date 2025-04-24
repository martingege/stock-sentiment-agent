# Stock Sentiment Agent

This project is designed to build an agentic system that pulls news articles and analyzes the sentiment regarding company stocks using LangChain.

## Project Structure

```
stock-sentiment-agent
├── src
│   ├── agents
│   │   ├── __init__.py
│   │   ├── news_agent.py
│   │   └── sentiment_agent.py
│   ├── tools
│   │   ├── __init__.py
│   │   ├── news_fetcher.py
│   │   └── sentiment_analyzer.py
│   └── main.py
├── tests
│   └── __init__.py
├── .env
├── requirements.txt
└── README.md
```

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd stock-sentiment-agent
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   Create a `.env` file in the root directory and add your API keys for news and sentiment analysis services.

## Usage

To run the application, execute the following command:

```bash
python src/main.py
```

This will initialize the agents, fetch news articles related to a specified company, and analyze the sentiment of those articles.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.