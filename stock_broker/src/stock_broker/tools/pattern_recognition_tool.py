import pandas as pd
import numpy as np
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Dict, List, Tuple

class PatternRecognitionInput(BaseModel):
    ohlcv_data: dict = Field(..., description="OHLCV market data from MarketDataTool")
    timeframe: str = Field("1min", description="Timeframe for pattern analysis")

class PatternRecognitionTool(BaseTool):
    name: str = "pattern_recognition_tool"
    description: str = "Identify chart patterns, support/resistance levels, and price action signals"
    args_schema: type[BaseModel] = PatternRecognitionInput

    def _run(self, ohlcv_data: dict, timeframe: str = "1min") -> dict:
        if not ohlcv_data:
            return {"error": "No OHLCV data provided"}

        # Convert to DataFrame
        df = pd.DataFrame.from_dict(ohlcv_data, orient='index')
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        results = {
            "patterns": [],
            "support_resistance": {},
            "price_action": {},
            "trend_analysis": {},
            "breakout_signals": []
        }

        try:
            # Identify chart patterns
            results["patterns"] = self._identify_patterns(df)
            
            # Calculate support and resistance levels
            results["support_resistance"] = self._find_support_resistance(df)
            
            # Analyze price action
            results["price_action"] = self._analyze_price_action(df)
            
            # Determine trend
            results["trend_analysis"] = self._analyze_trend(df)
            
            # Detect breakout signals
            results["breakout_signals"] = self._detect_breakouts(df, results["support_resistance"])

        except Exception as e:
            results["error"] = f"Pattern recognition failed: {str(e)}"

        return results

    def _identify_patterns(self, df: pd.DataFrame) -> List[Dict]:
        """Identify common chart patterns"""
        patterns = []
        
        if len(df) < 20:
            return patterns

        # Double Top/Bottom Pattern
        double_pattern = self._detect_double_top_bottom(df)
        if double_pattern:
            patterns.append(double_pattern)

        # Head and Shoulders
        h_and_s = self._detect_head_shoulders(df)
        if h_and_s:
            patterns.append(h_and_s)

        # Triangle Patterns
        triangle = self._detect_triangle(df)
        if triangle:
            patterns.append(triangle)

        # Flag/Pennant
        flag = self._detect_flag_pennant(df)
        if flag:
            patterns.append(flag)

        return patterns

    def _detect_double_top_bottom(self, df: pd.DataFrame) -> Dict:
        """Detect double top/bottom patterns"""
        highs = df['high'].rolling(window=5).max()
        lows = df['low'].rolling(window=5).min()
        
        # Find local maxima and minima
        local_maxima = (df['high'] == highs) & (df['high'].shift(1) < df['high']) & (df['high'].shift(-1) < df['high'])
        local_minima = (df['low'] == lows) & (df['low'].shift(1) > df['low']) & (df['low'].shift(-1) > df['low'])
        
        maxima_points = df[local_maxima]['high'].tail(3)
        minima_points = df[local_minima]['low'].tail(3)
        
        if len(maxima_points) >= 2:
            last_two_highs = maxima_points.tail(2)
            if abs(last_two_highs.iloc[0] - last_two_highs.iloc[1]) / last_two_highs.mean() < 0.02:  # Within 2%
                return {
                    "pattern": "Double Top",
                    "type": "bearish",
                    "confidence": 0.7,
                    "price_levels": [float(last_two_highs.iloc), float(last_two_highs.iloc[1])],
                    "signal": "Potential reversal - Consider selling"
                }
        
        if len(minima_points) >= 2:
            last_two_lows = minima_points.tail(2)
            if abs(last_two_lows.iloc - last_two_lows.iloc[1]) / last_two_lows.mean() < 0.02:  # Within 2%
                return {
                    "pattern": "Double Bottom", 
                    "type": "bullish",
                    "confidence": 0.7,
                    "price_levels": [float(last_two_lows.iloc), float(last_two_lows.iloc[1])],
                    "signal": "Potential reversal - Consider buying"
                }
        
        return None

    def _detect_head_shoulders(self, df: pd.DataFrame) -> Dict:
        """Detect head and shoulders pattern"""
        if len(df) < 30:
            return None
            
        # Simplified H&S detection - look for three peaks
        highs = df['high'].rolling(window=5).max()
        local_maxima = (df['high'] == highs) & (df['high'].shift(1) < df['high']) & (df['high'].shift(-1) < df['high'])
        peaks = df[local_maxima]['high'].tail(3)
        
        if len(peaks) == 3:
            left_shoulder, head, right_shoulder = peaks.iloc[0], peaks.iloc[1], peaks.iloc[2]
            
            # Check if middle peak (head) is higher than shoulders
            if head > left_shoulder and head > right_shoulder:
                # Check if shoulders are roughly equal (within 3%)
                if abs(left_shoulder - right_shoulder) / ((left_shoulder + right_shoulder) / 2) < 0.03:
                    return {
                        "pattern": "Head and Shoulders",
                        "type": "bearish",
                        "confidence": 0.8,
                        "price_levels": [float(left_shoulder), float(head), float(right_shoulder)],
                        "signal": "Strong reversal signal - Consider selling"
                    }
        
        return None

    def _detect_triangle(self, df: pd.DataFrame) -> Dict:
        """Detect triangle patterns (ascending, descending, symmetrical)"""
        if len(df) < 20:
            return None
            
        recent_df = df.tail(20)
        
        # Find trend lines for highs and lows
        high_trend = self._calculate_trendline(recent_df.index, recent_df['high'])
        low_trend = self._calculate_trendline(recent_df.index, recent_df['low'])
        
        if high_trend and low_trend:
            high_slope, low_slope = high_trend[1], low_trend[1]
            
            # Ascending triangle: flat resistance, rising support
            if abs(high_slope) < 0.001 and low_slope > 0.001:
                return {
                    "pattern": "Ascending Triangle",
                    "type": "bullish",
                    "confidence": 0.6,
                    "signal": "Bullish breakout expected"
                }
            
            # Descending triangle: declining resistance, flat support  
            elif high_slope < -0.001 and abs(low_slope) < 0.001:
                return {
                    "pattern": "Descending Triangle",
                    "type": "bearish", 
                    "confidence": 0.6,
                    "signal": "Bearish breakdown expected"
                }
            
            # Symmetrical triangle: converging lines
            elif high_slope < -0.001 and low_slope > 0.001:
                return {
                    "pattern": "Symmetrical Triangle",
                    "type": "neutral",
                    "confidence": 0.5,
                    "signal": "Breakout in either direction expected"
                }
        
        return None

    def _detect_flag_pennant(self, df: pd.DataFrame) -> Dict:
        """Detect flag and pennant patterns"""
        if len(df) < 15:
            return None
            
        # Look for strong move followed by consolidation
        recent_df = df.tail(15)
        price_change = (recent_df['close'].iloc[-1] - recent_df['close'].iloc[0]) / recent_df['close'].iloc
        
        # Strong initial move (>2%)
        if abs(price_change) > 0.02:
            # Check for consolidation (low volatility in recent periods)
            recent_volatility = recent_df['high'].rolling(3).max() - recent_df['low'].rolling(3).min()
            avg_volatility = recent_volatility.mean()
            
            if avg_volatility < recent_df['close'].mean() * 0.01:  # Low volatility
                pattern_type = "bullish" if price_change > 0 else "bearish"
                return {
                    "pattern": "Flag/Pennant",
                    "type": pattern_type,
                    "confidence": 0.6,
                    "signal": f"Continuation pattern - {pattern_type} move expected"
                }
        
        return None

    def _calculate_trendline(self, x_values, y_values) -> Tuple[float, float]:
        """Calculate trendline using linear regression"""
        try:
            x_numeric = np.arange(len(x_values))
            coeffs = np.polyfit(x_numeric, y_values, 1)
            return float(coeffs[1]), float(coeffs[0])  # intercept, slope
        except:
            return None

    def _find_support_resistance(self, df: pd.DataFrame) -> Dict:
        """Identify key support and resistance levels"""
        if len(df) < 10:
            return {}
            
        # Find pivot highs and lows
        window = 5
        df_copy = df.copy()
        
        # Pivot highs (resistance)
        df_copy['pivot_high'] = df_copy['high'][(df_copy['high'].shift(window) < df_copy['high']) & 
                                                (df_copy['high'].shift(-window) < df_copy['high'])]
        
        # Pivot lows (support)
        df_copy['pivot_low'] = df_copy['low'][(df_copy['low'].shift(window) > df_copy['low']) & 
                                              (df_copy['low'].shift(-window) > df_copy['low'])]
        
        # Get recent pivots
        resistance_levels = df_copy['pivot_high'].dropna().tail(3).tolist()
        support_levels = df_copy['pivot_low'].dropna().tail(3).tolist()
        
        current_price = float(df['close'].iloc[-1])
        
        # Find nearest levels
        nearest_resistance = min([r for r in resistance_levels if r > current_price], default=None)
        nearest_support = max([s for s in support_levels if s < current_price], default=None)
        
        return {
            "current_price": current_price,
            "nearest_resistance": nearest_resistance,
            "nearest_support": nearest_support,
            "all_resistance": resistance_levels,
            "all_support": support_levels,
            "resistance_strength": len(resistance_levels),
            "support_strength": len(support_levels)
        }

    def _analyze_price_action(self, df: pd.DataFrame) -> Dict:
        """Analyze recent price action signals"""
        if len(df) < 10:
            return {}
            
        recent = df.tail(10)
        current_price = float(df['close'].iloc[-1])
        
        # Calculate price action metrics
        body_size = abs(recent['close'] - recent['open']) / recent['open']
        wick_size = (recent['high'] - recent[['open', 'close']].max(axis=1)) + \
                   (recent[['open', 'close']].min(axis=1) - recent['low'])
        
        avg_body = body_size.mean()
        avg_wick = wick_size.mean()
        
        # Detect doji candles (small body)
        doji_threshold = 0.002  # 0.2%
        recent_doji = body_size.tail(3) < doji_threshold
        
        # Detect hammer/shooting star
        last_candle = recent.iloc[-1]
        last_body = abs(last_candle['close'] - last_candle['open'])
        last_upper_wick = last_candle['high'] - max(last_candle['open'], last_candle['close'])
        last_lower_wick = min(last_candle['open'], last_candle['close']) - last_candle['low']
        
        signals = []
        
        if recent_doji.any():
            signals.append("Doji pattern detected - Indecision in market")
            
        if last_lower_wick > 2 * last_body and last_upper_wick < last_body:
            signals.append("Hammer pattern - Potential bullish reversal")
            
        if last_upper_wick > 2 * last_body and last_lower_wick < last_body:
            signals.append("Shooting star - Potential bearish reversal")
        
        return {
            "avg_body_size": float(avg_body),
            "avg_wick_size": float(avg_wick),
            "recent_signals": signals,
            "market_sentiment": "volatile" if avg_wick > avg_body else "trending"
        }

    def _analyze_trend(self, df: pd.DataFrame) -> Dict:
        """Determine overall trend direction and strength"""
        if len(df) < 20:
            return {}
            
        # Calculate multiple timeframe trends
        short_term = df.tail(5)['close']
        medium_term = df.tail(10)['close'] 
        long_term = df.tail(20)['close']
        
        short_trend = (short_term.iloc[-1] - short_term.iloc[0]) / short_term.iloc
        medium_trend = (medium_term.iloc[-1] - medium_term.iloc) / medium_term.iloc
        long_trend = (long_term.iloc[-1] - long_term.iloc) / long_term.iloc
        
        # Determine trend direction
        def get_trend_direction(change):
            if change > 0.005:  # 0.5%
                return "bullish"
            elif change < -0.005:
                return "bearish"
            else:
                return "sideways"
        
        return {
            "short_term": {
                "direction": get_trend_direction(short_trend),
                "strength": abs(float(short_trend))
            },
            "medium_term": {
                "direction": get_trend_direction(medium_trend),
                "strength": abs(float(medium_trend))
            },
            "long_term": {
                "direction": get_trend_direction(long_trend),
                "strength": abs(float(long_trend))
            },
            "overall_trend": get_trend_direction((short_trend + medium_trend + long_trend) / 3)
        }

    def _detect_breakouts(self, df: pd.DataFrame, support_resistance: Dict) -> List[Dict]:
        """Detect recent breakouts from support/resistance levels"""
        if not support_resistance or len(df) < 5:
            return []
            
        current_price = support_resistance.get("current_price", 0)
        resistance_levels = support_resistance.get("all_resistance", [])
        support_levels = support_resistance.get("all_support", [])
        
        breakouts = []
        recent_high = df.tail(5)['high'].max()
        recent_low = df.tail(5)['low'].min()
        
        # Check resistance breakouts
        for resistance in resistance_levels:
            if recent_high > resistance and current_price > resistance:
                breakouts.append({
                    "type": "resistance_breakout",
                    "level": float(resistance),
                    "direction": "bullish",
                    "confidence": 0.7,
                    "signal": f"Price broke above resistance at {resistance:.2f}"
                })
        
        # Check support breakdowns
        for support in support_levels:
            if recent_low < support and current_price < support:
                breakouts.append({
                    "type": "support_breakdown", 
                    "level": float(support),
                    "direction": "bearish",
                    "confidence": 0.7,
                    "signal": f"Price broke below support at {support:.2f}"
                })
        
        return breakouts
