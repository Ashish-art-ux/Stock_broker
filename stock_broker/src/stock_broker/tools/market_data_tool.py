import os 
# from sqlite3 import _Parameters
# from validatory import validation
from stock_broker.utils.validators import validate_series
import time
import requests
from requests.exceptions import HTTPError
from tenacity import retry, wait_fixed,stop_after_attempt
# the CrewAI base class for all tools-handles argument parsing, logging, etc
from crewai.tools import BaseTool
# define and validate tool input parameters with typed schemas
from pydantic import BaseModel, Field

class MarketDataInput(BaseModel):
    # symbol: required string (e.g.,"AAPL")
    symbol: str = Field(..., description = "ticker symbol, e.g. AAPL")
    #interval:optional string defaulting to "1min"
    interval: str = Field("1min", description = "Time interval for intraday data")

class MarketDataTool(BaseTool):
    # the name field is simply the label agents use to call your tool.
    name: str =  "market_data_tool"
    #the description is a one-sentence summary of what your tool does
    description: str = "Fetch real-time and intraday market data"
    #args_schema is a pydantic model that lists exactly what information
    #your tool requires
    # in our case, it needs:
    # . symbol (the stock ticker, like "AAPL")
    # . interval (how often to sample price, default "1min")
    args_schema: type[BaseModel] =  MarketDataInput
    # role:Encapsulate your business logic: crewai does not touch or inspect its contents:
    # it simply:
    # 1 passes the validated arguments as _Parameters
    # 2 Executes _run()
    # 3 capture and return the result(or propagates exceptions to the agents)
    @retry(wait = wait_fixed(5), stop = stop_after_attempt(3))
    def fetch(self,url,params):
        resp = requests.get(url, params = params, timeout = 10)
        resp.raise_for_status()
        return resp.json()
    def _run(self, symbol: str, interval: str) -> dict:
        api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        url = "https://www.alphavantage.co/query"

        params = {
            "function" : "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": interval,
            "apikey": api_key,
        }

        time.sleep(12)
        data = self.fetch(url, params)

        # resp = requests.get(url, params = params)
        # resp.raise_for_status()
        # data = resp.json()
        series_key = f"Time Series ({interval})"
        valid = validate_series(data.get(series_key, {}))
        return valid
        # return data.get(series_key, {})