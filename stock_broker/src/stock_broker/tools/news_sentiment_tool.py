import os 
import requests
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import List, Dict

class NewsSentimentInput(BaseModel):
    symbol: str = Field(..., description = "Stock symbol to analyze news for")
    days: int = Field(3, description = "Number of days to look back for news")

class NewsSentimentTool(BaseTool):
    name: str = "news_sentiment_tool"
    description: str = "Analyze news sentiment for a stock symbol"
    args_schema: type[BaseModel] = NewsSentimentInput


    def _run(self, symbol: str, days: int = 3) -> dict:
        api_key = os.getenv("NEWS_API_KEY")
        if not api_key:
            return {"erro": "NEWS_API_KEY not found in enviroment"}

        #Search for news about the symbol
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": f"{symbol} stock OR {symbol} shares OR {self._get_company_name(symbol)}",
            "sortBy": "publisheAt",
            "language": "en",
            "apiKey": api_key,
            "pageSize":20
        }

        try:
            response = requests.get(url, params = params, timeout = 10)
            response.raise_for_status()
            data = response.json()

            articles = data.get("articles", [])
            if not articles:
                return {"sentiment": "neutral", "confidence": 0, "articles_count":0}


            #Analyze sentiment of headlines and descriptions 
            sentiment_scores = []
            for article in articles:
                title = article.get("title", "")
                description = article.get("discription", "")
                text = f"{title} {description}"
                score = self._analyze_text_sentiment(text)
                sentiment_scores.append(score)

            #Calculate overall sentiment
            avg_sentiment = sum(sentiment_scores)/len(sentiment_scores)
            sentiment_label = self._get_sentiment_label(avg_sentiment)

            return{
                "sentiment": sentiment_label,
                "confidence":abs(avg_sentiment),
                "articles_count": len(articles),
                "sentiment_score": avg_sentiment,
                "recent_headlines": [a.get("title", "")for a in articles[:5]]
            }
        except Exception as e:
            return {"error":F"News sentiment analysis failed: {str(e)}"}
    def _analyze_text_sentiment(self, text:str) ->float:
        """Simple sentiment analysis using keyword matching"""
        positive_words = ["buy", "bull", "bullish", "growth",
         "profit", "gain", "rise", "surge", "strong", "beat",
          "exceed", "positive", "upgrade", "outperform"]

        negative_words = ["sell", "bear", "bearish", 
        "loss", "decline", "fall", "drop", "weak",
         "miss", "disappoint", "negative",
          "downgrade", "underperform"]

        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count + negative_count == 0:
            return 0.0
        
        return (positive_count - negative_count) / (positive_count + negative_count)

    def _get_sentiment_label(self, score: float) -> str:
        """Convert numeric score to sentiment label"""
        if score > 0.2:
            return "positive"
        elif score < -0.2:
            return "negative"
        else:
            return "neutral"

    def _get_company_name(self, symbol: str) -> str:
        """Map common symbols to company names for better news search"""
        symbol_map = {
            "AAPL": "Apple",
            "GOOGL": "Google",
            "MSFT": "Microsoft",
            "TSLA": "Tesla",
            "AMZN": "Amazon",
            "META": "Meta",
            "NVDA": "NVIDIA"
        }
        return symbol_map.get(symbol.upper(), symbol)
