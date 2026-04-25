from newsapi import NewsApiClient
from backend.data.client import tiingo_client
from dotenv import load_dotenv
import os

load_dotenv()

NEWS_API_KEY = os.getenv("NEWS_API_KEY")

newsapi = NewsApiClient(api_key=NEWS_API_KEY)


ticker_metadata = tiingo_client.get_ticker_metadata("AAPL")
ticker_name = ticker_metadata.get("name")

response = newsapi.get_everything(
    q="GOOGL",
    from_param="2026-04-23",
    language="en",
)


titles = []
for article in response.get("articles", []):
    title = article.get("title")
    if title:
        titles.append(title)

print(titles)
