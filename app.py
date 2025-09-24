# File: app.py
from flask import Flask, render_template, jsonify, request
import yfinance as yf
from ta.volatility import BollingerBands
from ta.trend import SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator
import pandas as pd
import numpy as np
import pytz
import traceback

app = Flask(__name__)

SYMBOLS = {'ES': 'ES=F', 'SPX': '^GSPC'}
LOOKBACKS = {
    '1w': ('15m', '7d'),
    '1d': ('5m', '1d'),
    '1h': ('1m', '60m')
}

# --- Data fetching ---
def fetch_data(symbol, interval, period):
    # For ES, use a longer period and coarser interval for more data
    if symbol == 'ES=F' and interval == '5m' and period == '1d':
        interval = '15m'
        period = '5d'
    df = yf.download(tickers=symbol, interval=interval, period=period,
                     auto_adjust=False, progress=False)
    df.dropna(inplace=True)
    # convert to US/Eastern for session-based slicing
    if df.index.tzinfo is None:
        df.index = df.index.tz_localize('UTC')
    df.index = df.index.tz_convert('US/Eastern')
    return df

# --- Analysis functions ---
def add_indicators(df):
    close = df['Close']
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    df['SMA20'] = SMAIndicator(close, window=20).sma_indicator()
    df['EMA20'] = EMAIndicator(close, window=20).ema_indicator()
    df['RSI14'] = RSIIndicator(close, window=14).rsi()
    bb = BollingerBands(close, window=20)
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()
    df['VWAP'] = (df['Close'] * df['Volume']).cumsum() / df['Volume'].cumsum()
    return df

# Initial Balance (First Hour)
def initial_balance(df):
    # use 09:30-10:30 Eastern session
    fh = df.between_time('09:30', '10:30')
    if fh.empty:
        # Fallback: estimate frequency or default to 5 minutes
        freq = df.index.freq
        if freq is None:
            # Try to infer frequency from index
            if len(df.index) > 1:
                freq = df.index[1] - df.index[0]
            else:
                freq = pd.Timedelta('5min')
        else:
            freq = pd.Timedelta(freq)
        try:
            n = int(60 / (freq.seconds / 60))
        except Exception:
            n = 12  # Default to 12 bars (5min intervals)
        fh = df.head(n)
    high = fh['High'].max()
    low = fh['Low'].min()
    return float(high), float(low)

# Support/Resistance over last N bars
def support_resistance(df, window=100):
    recent = df.tail(window)
    return float(recent['Low'].min()), float(recent['High'].max())

# Detect Hammer Pattern in First Hour
def detect_patterns(df):
    pat = []
    fh = df.between_time('09:30', '10:30')
    if fh.empty or fh.isnull().all().all():
        print("Pattern detection: first hour data is empty")
        return []
    for t,r in fh.iterrows():
        try:
            body = abs(float(r['Open']) - float(r['Close']))
            rng = float(r['High']) - float(r['Low'])
            lower = min(float(r['Open']), float(r['Close'])) - float(r['Low'])
            # Relaxed hammer logic
            if rng > 0 and lower > 1.5*body and body < 0.5*rng:
                pat.append({'time': t.strftime('%H:%M'), 'pattern': 'Hammer'})
        except Exception as e:
            print(f"Pattern detection error: {e}")
            continue
    print(f"Pattern detection found {len(pat)} patterns")
    # Fallback: show first 3 bars if no patterns
    if not pat and not fh.empty:
        for t, r in fh.head(3).iterrows():
            pat.append({'time': t.strftime('%H:%M'), 'pattern': 'No pattern'})
    return pat

# Trend Prediction
def predict_trend(df):
    last = df.iloc[-1]
    c, s, rsi = float(last['Close']), float(last['SMA20']), float(last['RSI14'])
    return ('Up', 0.65) if (c > s and rsi > 50) else ('Down', 0.35)

# Price Range Forecast
def price_range(df, days):
    last = float(df['Close'].iloc[-1])
    std = df['Close'].rolling(20).std().iloc[-1]
    # Fix: handle Series or NaN std
    if isinstance(std, pd.Series):
        std = std.iloc[0]
    if pd.isna(std):
        std = 0.0
    return (last - std*days, last + std*days)

# Placeholder probabilities
prob_hit = lambda: np.random.randint(30,70)
prob_break = lambda: np.random.randint(20,50)

@app.route('/')
def dashboard():
    # Fetch and analyze SPX and ES
    spx = {key: add_indicators(fetch_data(SYMBOLS['SPX'], ivl, per))
           for key, (ivl, per) in LOOKBACKS.items()}
    es = {key: add_indicators(fetch_data(SYMBOLS['ES'], ivl, per))
          for key, (ivl, per) in LOOKBACKS.items()}

    # SPX metrics
    ib_high, ib_low = initial_balance(spx['1d'])
    sup, res = support_resistance(spx['1w'])
    patterns = detect_patterns(spx['1d'])
    trend_1d, prob1 = predict_trend(spx['1d'])
    r1_low, r1_high = price_range(spx['1d'], 1)

    # ES metrics
    es_ib_high, es_ib_low = initial_balance(es['1d'])
    es_sup, es_res = support_resistance(es['1w'])
    es_patterns = detect_patterns(es['1d'])
    es_trend_1d, es_prob1 = predict_trend(es['1d'])
    es_r1_low, es_r1_high = price_range(es['1d'], 1)

    return render_template('dashboard.html',
        ib_high=round(ib_high,2), ib_low=round(ib_low,2),
        vwap=round(spx['1d']['VWAP'].iloc[-1],2),
        support=round(sup,2), resistance=round(res,2),
        prob_hit=prob_hit(), prob_res=prob_break(),
        prob_break_res=prob_break(), prob_break_sup=prob_break(),
        trend_1d=trend_1d, prob1=int(prob1*100),
        range_low=round(r1_low,2), range_high=round(r1_high,2),
        patterns=patterns,
        # ES data
        es_ib_high=round(es_ib_high,2), es_ib_low=round(es_ib_low,2),
        es_vwap=round(es['1d']['VWAP'].iloc[-1],2),
        es_support=round(es_sup,2), es_resistance=round(es_res,2),
        es_prob_hit=prob_hit(), es_prob_res=prob_break(),
        es_prob_break_res=prob_break(), es_prob_break_sup=prob_break(),
        es_trend_1d=es_trend_1d, es_prob1=int(es_prob1*100),
        es_range_low=round(es_r1_low,2), es_range_high=round(es_r1_high,2),
        es_patterns=es_patterns
    )

@app.route('/api/bars')
def api_bars():
    try:
        bars = request.args.get('bars','1d')
        symbol = request.args.get('symbol','SPX')
        ivl, per = LOOKBACKS.get(bars, LOOKBACKS['1d'])
        df = add_indicators(fetch_data(SYMBOLS[symbol], ivl, per))
        out = df.reset_index()
        # Flatten columns if multi-indexed (yfinance sometimes returns this)
        out.columns = [
            col if isinstance(col, str) else col[0] if col[0] else col[1] if col[1] else str(col) for col in out.columns
        ]
        # Rename first column to 'Datetime' for consistency
        out.rename(columns={out.columns[0]:'Datetime'}, inplace=True)
        out['Datetime'] = pd.to_datetime(out['Datetime']).dt.strftime('%Y-%m-%dT%H:%M:%S')
        cols = ['Datetime','Close','SMA20','EMA20','BB_High','BB_Low','VWAP']
        # Fix: convert all keys to strings for jsonify and replace NaN with None
        result = []
        for row in out[cols].to_dict(orient='records'):
            clean_row = {str(k): (None if pd.isna(v) else v) for k, v in row.items()}
            result.append(clean_row)
        return jsonify(result)
    except Exception as e:
        print("API /api/bars error:", e)
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__=='__main__': 
    app.run(debug=True)