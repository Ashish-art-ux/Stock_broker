
def validate_series(series: dict):
    """Ensure each bar has proper OHLCV data from Alpha vintage format"""
    clean = {}
    for ts, bar in series.items():
        if not isinstance(bar, dict):
            continue
        try:
            clean[ts] = {
                'open': float(bar.get('1. open', 0)),
                'high': float(bar.get('2.high', 0)),
                'low': float(bar.get('3. low', 0)),
                'close': float(bar.get('4. close', 0)),
                'volume': float(bar.get('5. volume', 0))
            }
        except (ValueError, TypeError, KeyError):
            continue
    return clean
    