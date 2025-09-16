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
symbols = { 'SPX': '^GSPC'}

data = {}
for sym, ticker in symbols.items():
    data[sym] = {

        '1h': get_data(ticker, interval='15m', period = '1d')
        }

print(data)
print(SMAIndicator(data["SPX"]["1h"]["Close"].squeeze(), window=10).sma_indicator())

def add_indicators(df):
    close = df['Close'].squeeze()
    df['SMA20'] = SMAIndicator(close, window=20).sma_indicator()
    df['EMA20'] = EMAIndicator(close, window=20).ema_indicator()
    df['RSI14'] = RSIIndicator(close, window=14).rsi()
    bb = BollingerBands(close, window=20)
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()
    df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    return df

for sym in data:
    for tf in data[sym]:
        data[sym][tf] = add_indicators(data[sym][tf])

print(data)


def calculate_vwap(df):
    vwap = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    df['VWAP'] = vwap
    return df

for tf in ['1h']:
    data['SPX'][tf] = calculate_vwap(data['SPX'][tf])

def initial_balance(df):
    first_hour = df.between_time('13:30', '14:30')#convert to UTC
    ib_high = first_hour['High'].max().item()
    ib_low = first_hour['Low'].min().item()
    return ib_high, ib_low

ib_high, ib_low = initial_balance(data['SPX']['1h'])#1d
print(f"Initial Balance High: {ib_high}, Initial Balance Low: {ib_low}")


def find_levels(df):
    recent = df.tail(100)
    support = recent['Low'].min().item()
    resistance = recent['High'].max().item()
    return support, resistance

support, resistance = find_levels(data['SPX']['1h'])#1w
print(f"Support: {support}, Resistance: {resistance}")


def detect_candlestick_patterns(df):
# Simple hammer pattern example
    first_hour = df.between_time('13:30', '14:30')
    patterns = []
    print(first_hour)
    for idx, row in first_hour.iterrows():
        print(row)
        print("HIHIHIHIHIH")
        '''body = abs(row['Open'] - row['Close'])
        range_ = row['High'] - row['Low']
        lower_wick = min(row['Open'], row['Close']) - row['Low']
        if lower_wick > 2 * body and body < 0.3 * range_:
            patterns.append((idx, 'Hammer'))
            '''
    return patterns
detect_candlestick_patterns(data['SPX']['1h'])
'''
patterns = detect_candlestick_patterns(data['SPX']['1h'])
print(f"Candlestick Patterns: {patterns}")'''