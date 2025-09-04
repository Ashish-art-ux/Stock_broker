import os
import time
import requests
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Dict, List
from stock_broker.utils.validators import validate_series

class MultiTimeframeInput(BaseModel):
    symbol: str = Field(..., description="Stock symbol to analyze")
    timeframes: List[str] = Field(["1min", "5min", "15min", "60min"], description="List of timeframes")

class MultiTimeframeTool(BaseTool):
    name: str = "multi_timeframe_tool"
    description: str = "Analyze multiple timeframes to confirm trends and signals"
    args_schema: type[BaseModel] = MultiTimeframeInput

    def _run(self, symbol: str, timeframes: List[str]) -> Dict:
        api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        if not api_key:
            return {"error": "ALPHA_VANTAGE_API_KEY not found"}

        results = {
            "symbol": symbol,
            "timeframe_analysis": {},
            "trend_alignment": {},
            "signal_confluence": [],
            "recommendation": ""
        }

        try:
            # Fetch data for each timeframe
            for tf in timeframes:
                tf_data = self._fetch_timeframe_data(symbol, tf, api_key)
                if tf_data:
                    results["timeframe_analysis"][tf] = self._analyze_timeframe(tf_data, tf)
                
                # Rate limiting
                time.sleep(12)
            
            # Analyze trend alignment across timeframes
            results["trend_alignment"] = self._analyze_trend_alignment(results["timeframe_analysis"])
            
            # Find signal confluence
            results["signal_confluence"] = self._find_signal_confluence(results["timeframe_analysis"])
            
            # Generate multi-timeframe recommendation
            results["recommendation"] = self._generate_mtf_recommendation(results)

        except Exception as e:
            results["error"] = f"Multi-timeframe analysis failed: {str(e)}"

        return results

    def _fetch_timeframe_data(self, symbol: str, interval: str, api_key: str) -> Dict:
        """Fetch data for specific timeframe"""
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": interval,
            "apikey": api_key,
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            series_key = f"Time Series ({interval})"
            raw_series = data.get(series_key, {})
            return validate_series(raw_series)
            
        except Exception as e:
            print(f"Error fetching {interval} data: {str(e)}")
            return {}

    def _analyze_timeframe(self, data: Dict, timeframe: str) -> Dict:
        """Analyze single timeframe data"""
        if not data:
            return {}
            
        import pandas as pd
        df = pd.DataFrame.from_dict(data, orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        
        if len(df) < 10:
            return {}

        # Calculate basic metrics
        current_price = float(df['close'].iloc[-1])
        price_change = (current_price - df['close'].iloc[0]) / df['close'].iloc
        
        # Simple trend analysis
        sma_short = df['close'].rolling(window=min(5, len(df))).mean().iloc[-1]
        sma_long = df['close'].rolling(window=min(10, len(df))).mean().iloc[-1]
        
        trend = "bullish" if current_price > sma_short > sma_long else \
               "bearish" if current_price < sma_short < sma_long else "sideways"
        
        # Volume analysis (if available)
        avg_volume = df['volume'].mean() if 'volume' in df.columns else 0
        recent_volume = df['volume'].tail(3).mean() if 'volume' in df.columns else 0
        volume_trend = "increasing" if recent_volume > avg_volume * 1.1 else \
                      "decreasing" if recent_volume < avg_volume * 0.9 else "stable"
        
        return {
            "timeframe": timeframe,
            "current_price": current_price,
            "price_change_pct": float(price_change * 100),
            "trend": trend,
            "sma_short": float(sma_short),
            "sma_long": float(sma_long),
            "volume_trend": volume_trend,
            "data_points": len(df)
        }

    def _analyze_trend_alignment(self, timeframe_data: Dict) -> Dict:
        """Analyze trend alignment across timeframes"""
        if not timeframe_data:
            return {}
            
        trends = {}
        bullish_count = 0
        bearish_count = 0
        sideways_count = 0
        
        for tf, data in timeframe_data.items():
            if 'trend' in data:
                trend = data['trend']
                trends[tf] = trend
                
                if trend == "bullish":
                    bullish_count += 1
                elif trend == "bearish":
                    bearish_count += 1
                else:
                    sideways_count += 1
        
        total_timeframes = len(trends)
        if total_timeframes == 0:
            return {}
        
        # Determine overall alignment
        if bullish_count >= total_timeframes * 0.75:
            alignment = "strongly_bullish"
        elif bearish_count >= total_timeframes * 0.75:
            alignment = "strongly_bearish"
        elif bullish_count > bearish_count:
            alignment = "moderately_bullish"
        elif bearish_count > bullish_count:
            alignment = "moderately_bearish"
        else:
            alignment = "mixed"
        
        return {
            "trends": trends,
            "bullish_count": bullish_count,
            "bearish_count": bearish_count,
            "sideways_count": sideways_count,
            "total_timeframes": total_timeframes,
            "alignment": alignment,
            "alignment_strength": max(bullish_count, bearish_count) / total_timeframes
        }

    def _find_signal_confluence(self, timeframe_data: Dict) -> List[Dict]:
        """Find confluent signals across timeframes"""
        signals = []
        
        # Check for trend confluence
        bullish_tfs = [tf for tf, data in timeframe_data.items() 
                      if data.get('trend') == 'bullish']
        bearish_tfs = [tf for tf, data in timeframe_data.items() 
                      if data.get('trend') == 'bearish']
        
        if len(bullish_tfs) >= 3:
            signals.append({
                "type": "trend_confluence",
                "direction": "bullish",
                "timeframes": bullish_tfs,
                "strength": len(bullish_tfs) / len(timeframe_data),
                "signal": f"Bullish trend confirmed across {len(bullish_tfs)} timeframes"
            })
        
        if len(bearish_tfs) >= 3:
            signals.append({
                "type": "trend_confluence", 
                "direction": "bearish",
                "timeframes": bearish_tfs,
                "strength": len(bearish_tfs) / len(timeframe_data),
                "signal": f"Bearish trend confirmed across {len(bearish_tfs)} timeframes"
            })
        
        # Check for volume confluence
        increasing_volume_tfs = [tf for tf, data in timeframe_data.items()
                                if data.get('volume_trend') == 'increasing']
        
        if len(increasing_volume_tfs) >= 2:
            signals.append({
                "type": "volume_confluence",
                "direction": "bullish",
                "timeframes": increasing_volume_tfs,
                "strength": len(increasing_volume_tfs) / len(timeframe_data),
                "signal": f"Increasing volume across {len(increasing_volume_tfs)} timeframes"
            })
        
        return signals

    def _generate_mtf_recommendation(self, results: Dict) -> str:
        """Generate recommendation based on multi-timeframe analysis"""
        alignment = results.get("trend_alignment", {})
        confluence = results.get("signal_confluence", [])
        
        if not alignment:
            return "Insufficient data for multi-timeframe analysis"
        
        alignment_type = alignment.get("alignment", "mixed")
        alignment_strength = alignment.get("alignment_strength", 0)
        
        recommendation = f"Multi-Timeframe Analysis Summary:\n"
        recommendation += f"Trend Alignment: {alignment_type.replace('_', ' ').title()}\n"
        recommendation += f"Alignment Strength: {alignment_strength:.1%}\n\n"
        
        if alignment_type == "strongly_bullish":
            recommendation += "STRONG BUY: All timeframes show bullish alignment\n"
            recommendation += "High confidence for upward movement\n"
        elif alignment_type == "strongly_bearish":
            recommendation += "STRONG SELL: All timeframes show bearish alignment\n"
            recommendation += "High confidence for downward movement\n"
        elif alignment_type in ["moderately_bullish", "moderately_bearish"]:
            direction = "BUY" if "bullish" in alignment_type else "SELL"
            recommendation += f"MODERATE {direction}: Majority of timeframes aligned\n"
            recommendation += "Medium confidence, monitor for confirmation\n"
        else:
            recommendation += "HOLD: Mixed signals across timeframes\n"
            recommendation += "Wait for clearer directional bias\n"
        
        if confluence:
            recommendation += f"\nConfluent Signals Detected:\n"
            for signal in confluence:
                recommendation += f"- {signal['signal']}\n"
        
        return recommendation
