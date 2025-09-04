# test_apis.py
import os
from dotenv import load_dotenv
import requests

load_dotenv()

# Test Gemini API
gemini_key = os.getenv("GEMINI_API_KEY")
print(f"Gemini API Key: {gemini_key[:10]}..." if gemini_key else "Gemini API Key: Missing")

# Test Alpha Vantage
av_key = os.getenv("ALPHA_VANTAGE_API_KEY")
if av_key:
    response = requests.get(f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol=AAPL&interval=1min&apikey={av_key}")
    print(f"Alpha Vantage: {'✓ Working' if response.status_code == 200 else '✗ Error'}")

# Test News API
news_key = os.getenv("NEWS_API_KEY")
if news_key:
    response = requests.get(f"https://newsapi.org/v2/everything?q=stocks&apiKey={news_key}")
    print(f"News API: {'✓ Working' if response.status_code == 200 else '✗ Error'}")

# Test Twitter Bearer Token
twitter_token = os.getenv("TWITTER_BEARER_TOKEN")
if twitter_token:
    headers = {'Authorization': f'Bearer {twitter_token}'}
    response = requests.get('https://api.x.com/2/tweets/search/recent?query=AAPL', headers=headers)
    print(f"Twitter API: {'✓ Working' if response.status_code == 200 else '✗ Error'}")
