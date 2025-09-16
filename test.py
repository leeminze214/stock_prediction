import pandas as pd
import numpy as np
import yfinance as yf
from ta.volatility import BollingerBands
from ta.trend import SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator
from datetime import datetime, timedelta

# --- Data Retrieval ---
def get_data(symbol, interval, period):
    data = yf.download(tickers=symbol, interval=interval, period=period)
    return data

# --- Fetch Data ---
now = datetime.now()
symbols = {'ES': 'ES=F', 'SPX': '^GSPC'}

data = {}
for sym, ticker in symbols.items():
    data[sym] = {
        '1w': get_data(ticker, '15m', '7d'),
        '1d': get_data(ticker, '5m', '1d'),
        '1h': get_data(ticker, '1m', '60m')
        }

# --- Technical Indicators ---
def add_indicators(df):
    df['SMA20'] = SMAIndicator(df['Close'], window=20).sma_indicator()
    df['EMA20'] = EMAIndicator(df['Close'], window=20).ema_indicator()
    df['RSI14'] = RSIIndicator(df['Close'], window=14).rsi()
    bb = BollingerBands(df['Close'], window=20)
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()
    return df

for sym in data:
    for tf in data[sym]:
        data[sym][tf] = add_indicators(data[sym][tf])

# --- VWAP Calculation ---
def calculate_vwap(df):
    vwap = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    df['VWAP'] = vwap
    return df

for tf in ['1d', '1h']:
    data['SPX'][tf] = calculate_vwap(data['SPX'][tf])

# --- Initial Balance and First Hour High/Low ---
def initial_balance(df):
    first_hour = df.between_time('09:30', '10:30')
    ib_high = first_hour['High'].max()
    ib_low = first_hour['Low'].min()
    return ib_high, ib_low

ib_high, ib_low = initial_balance(data['SPX']['1d'])

# --- Support and Resistance ---
def find_levels(df):
    recent = df.tail(100)
    support = recent['Low'].min()
    resistance = recent['High'].max()
    return support, resistance

support, resistance = find_levels(data['SPX']['1w'])

# --- Candlestick Pattern Detection (First Hour) ---
def detect_candlestick_patterns(df):
# Simple hammer pattern example
    first_hour = df.between_time('09:30', '10:30')
    patterns = []
    for idx, row in first_hour.iterrows():
        body = abs(row['Open'] - row['Close'])
        range_ = row['High'] - row['Low']
        lower_wick = min(row['Open'], row['Close']) - row['Low']
        if lower_wick > 2 * body and body < 0.3 * range_:
            patterns.append((idx, 'Hammer'))
    return patterns

patterns = detect_candlestick_patterns(data['SPX']['1d'])

# --- Trend Prediction (Simple Example) ---
def predict_trend(df):
    # Example: if price > SMA20 and RSI > 50, uptrend; else, downtrend
    last = df.iloc[-1]
    if last['Close'] > last['SMA20'] and last['RSI14'] > 50:
        return 'Up', 0.65 # 65% probability (example)
    else:
        return 'Down', 0.35

trend_1d, prob_1d = predict_trend(data['SPX']['1d'])
trend_3d, prob_3d = predict_trend(data['SPX']['1w'])
trend_1w, prob_1w = predict_trend(data['SPX']['1w'])

# --- Price Range Prediction (Naive Example) ---
def price_range(df, days):
    last = df['Close'].iloc[-1]
    std = df['Close'].rolling(window=20).std().iloc[-1]
    return (last - std * days, last + std * days)

range_1d = price_range(data['SPX']['1d'], 1)
range_3d = price_range(data['SPX']['1w'], 3)
range_1w = price_range(data['SPX']['1w'], 5)

# --- Output ---
print("SPX Technical Analysis:")
print(f"- Initial Balance (First Hour High/Low): {ib_high:.2f} / {ib_low:.2f}")
print(f"- VWAP (1d): {data['SPX']['1d']['VWAP'].iloc[-1]:.2f}")
print(f"- Support: {support:.2f}, Resistance: {resistance:.2f}")
print(f"- Detected Candlestick Patterns (First Hour): {patterns}")
print(f"- 1D Trend: {trend_1d} ({prob_1d*100:.0f}%), Range: {range_1d}")
print(f"- 3D Trend: {trend_3d} ({prob_3d*100:.0f}%), Range: {range_3d}")
print(f"- 1W Trend: {trend_1w} ({prob_1w*100:.0f}%), Range: {range_1w}")

# Probabilities for support/resistance breaks (example, customize as needed)
print(f"- Probability of hitting support: 40%")
print(f"- Probability of hitting resistance: 60%")
print(f"- Probability of breaking above resistance: 30%")
print(f"- Probability of breaking below support: 20%")