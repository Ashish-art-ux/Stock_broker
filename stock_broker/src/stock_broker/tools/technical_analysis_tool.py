import pandas as pd
import numpy as np
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import List, Dict, Any

class TechnicalAnalysisInput(BaseModel):
    ohlcv_data: dict = Field(..., description = "OHLCV market data from MarketDataTool")
    indicators: List[str] = Field(["rsi", "macd", "sma"],
    description = "Technical indicators to calculate")


class TechnicalAnalysisTool(BaseTool):
    name:str = "technical_analysis_tool"
    description: str = "calculate technical indicators (RSI, MACD, SMA, EMA) from OHLCV data"
    args_schema: type[BaseModel] = TechnicalAnalysisInput

    def _run(self, ohlcv_data: dict, indicators: List[str])->dict:
        if not ohlcv_data:
            return{"error": "No OHLCV data provided"}

        #convert to Dataframe
        df = pd.DataFrame.from_dict(ohlcv_data, orient = 'index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        results = {"indicators": {}, "signals": [],
        "analysis":""}

        try:
            #calculate RSI (Relative Strength Index)
            if "rsi" in [i.lower() for i in indicators]:
                results["indicators"]["rsi"] =  self._calculate_rsi(df['close'])


            if "macd" in [i.lower() for i in indicators]:
                macd_data = self._calculate_macd(df['close'])
                results["indicators"]["macd"] = macd_data

            if "sma" in [i.lower() for i in indicators]:
                results["indicators"]["sum_20"] = self._calculate_sma(df['close'],20)
                results["indicators"]["sma_50"] = self._calculate_sma(df['close'],50)

            # Calculate Exponential Moving Average
            if "ema" in [i.lower() for i in indicators]:
                results["indicators"]["ema_12"] = self._calculate_ema(df['close'], 12)
                results["indicators"]["ema_26"] = self._calculate_ema(df['close'], 26)

            #Generate trading signals
            results["signals"] = self._generate_signals(df, results["indicators"])

            #Generate analysis summary
            results["analysis"] = self._generate_analysis(results["signals"], results["indicators"])

        except Exception as e:
            results["error"] = f"Technical analysis failed: {str(e)}"

        return results

    def _calculate_rsi(self, closes: pd.Series, period: int = 14) -> dict:
        """claculate Rsi indicator"""
        delta = closes.diff()
        gain = (delta.where(delta > 0,0)).rolling(window = period).mean()
        loss = (-delta.where(delta < 0,0)).rolling(window = period).mean()
        rs = gain / loss
        rsi = 100 - (100/(1+rs))

        current_rsi = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50
        return {
            "current": current_rsi,
            "signal": "oversold" if current_rsi <30 else "overbought" if current_rsi > 70 else "neutral"

        }
    def _calculate_macd(self, closes: pd.Series) ->dict:
        """Calculate MACD indicator"""
        ema12 = closes.ewm(span = 12).mean()
        ema26 = closes.ewm(span = 26).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span = 9).mean()
        histogram = macd_line - signal_line

        return {
            "macd": float(macd_line.iloc[-1]) if not pd.isna(macd_line.iloc[-1]) else 0,
            "signal": float(signal_line.iloc[-1]) if not pd.isna(signal_line.iloc[-1]) else 0,
            "histogram": float(histogram.iloc[-1]) if not pd.isna(histogram.iloc[-1]) else 0,
            "trend": "bullish" if histogram.iloc[-1] > 0 else "bearish"
        }

    def _calculate_sma(self, closes:pd.Series, period: int)->dict:
        """Calculate Simple Moving Average"""
        sma = closes.rolling(window=period).mean()
        current_price = float(closes.iloc[-1])
        current_sma = float(sma.iloc[-1]) if not pd.isna(sma.iloc[-1]) else current_price

        return {
            "value":current_sma,
            "position": "above" if current_price > current_sma else "below"

        }

    def _calculate_ema(self, closes: pd.Series, period: int)->dict:
        """Calculate Exponential Moving Average"""
        ema = closes.ewm(span = period).mean()
        current_price = float(closes.iloc[-1])
        current_ema = float(ema.iloc[-1]) if not pd.isna(ema.iloc[-1]) else current_price

        return{
            "value": current_ema,
            "position": "above" if current_price > current_ema else "below"
        }
    def _generate_signals(self, df: pd.DataFrame, indicators: dict) ->List[str]:
        """generate trading signald based on indicators"""
        signals = []

        #RSI signals
        if "rsi" in indicators:
            if indicators["rsi"]["signal"] == "oversold":
                signals.append("BUY: RSI indicates oversold condition")
            elif indicators["rsi"]["signal"]=="overbought":
                signals.append("SELL: RSI indicates overbought condition")

        #MACD signals
        if "macd" in indicators:
            if indicators["macd"]["trend"] =="bullish":
                signals.append("BUY: MACD showing bullish momentum")
            else:
                signals.append("SELL: MACD showing bearish momentum")
        if "sma_20" in indicators and "sma_50" in indicators:
            if indicators["sma_20"]["value"] > indicators["sma_50"]["value"]:
                signals.append("BUY: Short-term SMA above long-term SMA")
            else:
                signals.append("SELL: Short-term SMA below long-term SMA")
        
        return signals


    def _generate_analysis(self, signals: List[str], indicators: dict) -> str:
        """Generate comprehensive analysis summary"""
        buy_signals = len([s for s in signals if s.startswith("BUY")])
        sell_signals = len([s for s in signals if s.startswith("SELL")])
        
        if buy_signals > sell_signals:
            sentiment = "BULLISH"
        elif sell_signals > buy_signals:
            sentiment = "BEARISH"
        else:
            sentiment = "NEUTRAL"
            
        analysis = f"Technical Analysis Summary: {sentiment}\n"
        analysis += f"Buy signals: {buy_signals}, Sell signals: {sell_signals}\n"
        
        if "rsi" in indicators:
            analysis += f"RSI: {indicators['rsi']['current']:.2f} ({indicators['rsi']['signal']})\n"
        if "macd" in indicators:
            analysis += f"MACD: {indicators['macd']['trend']} trend\n"
            
        return analysis