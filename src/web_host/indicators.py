import yfinance as yf
import pandas_ta as ta
import pandas as pd

def fetch_stock_data(ticker):
    """
    Fetch historical price and volume data for a given ticker.
    """
    df = yf.download(ticker, period="3mo", interval="1d")
    return df

def calculate_moving_average(df, window=20):
    """
    Calculate the moving average (default 20 days).
    """
    return df['Close'].rolling(window=window).mean().iloc[-1]

def calculate_rsi(df, length=14):
    """
    Calculate the latest RSI value safely.
    """
    close_series = df['Close']
    if isinstance(close_series, pd.DataFrame):
        close_series = close_series.squeeze()  # flatten DataFrame to Series if needed

    rsi_series = ta.rsi(close_series, length=length)

    if rsi_series is None or rsi_series.isna().all():
        return None
    else:
        return rsi_series.dropna().iloc[-1]

def check_volume_spike(df, multiplier=1.5, window=10):
    """
    Check if the latest volume is significantly higher than the moving average volume.
    """
    avg_volume_series = df['Volume'].rolling(window=window).mean()
    latest_volume = df['Volume'].iloc[-1]
    avg_volume = avg_volume_series.iloc[-1]

    # ⚡ Force extraction of scalar value
    if hasattr(latest_volume, 'item'):
        latest_volume = latest_volume.item()
    if hasattr(avg_volume, 'item'):
        avg_volume = avg_volume.item()

    if not pd.isna(avg_volume) and not pd.isna(latest_volume):
        return latest_volume > multiplier * avg_volume, latest_volume, avg_volume
    else:
        return False, latest_volume, avg_volume    


def gather_indicators(ticker):
    """
    Fetch all three key indicators: Moving Average, RSI, Volume Spike.
    """
    df = fetch_stock_data(ticker)

    if df.empty:
        return None

    moving_avg = calculate_moving_average(df)
    rsi = calculate_rsi(df)

    # Force scalar values for current price and moving average
    current_price = df['Close'].iloc[-1]
    if hasattr(current_price, 'item'):
        current_price = current_price.item()
    if hasattr(moving_avg, 'item'):
        moving_avg = moving_avg.item()

    volume_spike, latest_volume, avg_volume = check_volume_spike(df)

    indicators = {
        'current_price': current_price,
        'moving_avg': moving_avg,
        'rsi': rsi,
        'volume_spike': volume_spike,
        'latest_volume': latest_volume,
        'avg_volume': avg_volume,
    }

    return indicators

def calculate_confidence(price, moving_avg, rsi, latest_volume, avg_volume):
    score = 0
    
    # Moving Average
    if price > moving_avg:
        score += 30
    
    # RSI
    if 40 < rsi < 65:
        score += 30
    elif rsi < 30 and price < moving_avg:
        score += 15  # Oversold + Downtrend = possible bounce
    
    # Volume
    if latest_volume > 1.5 * avg_volume:
        score += 40

    return score

def suggest_action_with_confidence(price, moving_avg, rsi, latest_volume, avg_volume):
    confidence = calculate_confidence(price, moving_avg, rsi, latest_volume, avg_volume)

    if price > moving_avg and 40 < rsi < 65 and latest_volume > 1.5 * avg_volume:
        action = "📈 Suggest: BUY"
    elif (price < moving_avg or rsi > 70) and latest_volume > 1.5 * avg_volume:
        action = "📉 Suggest: SELL"
    else:
        action = "⏳ Suggest: HOLD"
    
    return action, confidence


# Example usage:
if __name__ == "__main__":
    ticker = "AAPL"
    indicators = gather_indicators(ticker)
    print(indicators)
    suggest_action_with_confidence = suggest_action_with_confidence(
        indicators['current_price'],
        indicators['moving_avg'],
        indicators['rsi'],
        indicators['latest_volume'],
        indicators['avg_volume']
    )
    print(suggest_action_with_confidence)
    print(f"Action: {suggest_action_with_confidence[0]}, Confidence Score: {suggest_action_with_confidence[1]}")
