import time
import numpy as np
import pandas as pd
import yfinance as yf
try:
    import crypto_yfinance as cyf
except ImportError:
    crypto_yfinance = None
from PIL import Image
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, timedelta
from tensorflow.keras.optimizers import Adam
from streamlit_option_menu import option_menu
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.layers import Conv1D, GRU, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score, mean_absolute_error

# Helper function to safely extract scalar from pandas Series
def safe_float(value):
    if hasattr(value, 'item'):
        return float(value.item())
    return float(value)

def smart_format(val, default_decimals=2, prefix=""):
    """
    Format angka secara cerdas dengan dukungan sub-angka (subscript) untuk 0.000...
    - Jika val adalah None / NaN: kembalikan "-"
    - Jika val == 0: kembalikan prefix + "0.00"
    - Jika abs(val) >= 0.01: format standar (misal 1,234.56 atau 0.05)
    - Jika 0 < abs(val) < 0.01: format 0.0_{subscript}N... (misal 0.0₄1234 untuk 0.00001234)
    """
    if val is None or (isinstance(val, (float, np.floating, int, np.integer)) and pd.isna(val)):
        return "-"
    try:
        fval = float(val)
    except (ValueError, TypeError):
        return str(val)

    if fval == 0:
        return f"{prefix}0.00"

    sign = "-" if fval < 0 else ""
    abs_val = abs(fval)

    if abs_val >= 0.01:
        if abs_val >= 1000:
            return f"{prefix}{sign}{abs_val:,.2f}"
        elif default_decimals == 3:
            return f"{prefix}{sign}{abs_val:.3f}"
        elif default_decimals == 4:
            return f"{prefix}{sign}{abs_val:.4f}"
        else:
            return f"{prefix}{sign}{abs_val:.2f}"

    # Untuk nilai sangat kecil: 0 < abs_val < 0.01
    exp_str = f"{abs_val:.10e}"
    parts = exp_str.split('e')
    mantissa = float(parts[0])
    exponent = int(parts[1])

    zero_count = abs(exponent) - 1

    subscript_digits = {'0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄', '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉'}
    sub_str = ''.join(subscript_digits[d] for d in str(zero_count))

    sig_digits = f"{mantissa:.4f}".replace('.', '').rstrip('0')
    if not sig_digits:
        sig_digits = f"{mantissa:.2f}".replace('.', '')

    return f"{prefix}{sign}0.0{sub_str}{sig_digits}"

def flatten_df_columns(df):
    if df is None or df.empty:
        return df
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def ensure_datetime_index(df):
    if df is None or df.empty:
        return pd.DataFrame()
    df = flatten_df_columns(df)
    
    if isinstance(df.index, pd.DatetimeIndex):
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.index.name = 'Date'
        return df
        
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        if hasattr(df['Date'].dt, 'tz') and df['Date'].dt.tz is not None:
            df['Date'] = df['Date'].dt.tz_localize(None)
        df.set_index('Date', inplace=True)
    elif 'Datetime' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        if hasattr(df['Datetime'].dt, 'tz') and df['Datetime'].dt.tz is not None:
            df['Datetime'] = df['Datetime'].dt.tz_localize(None)
        df.set_index('Datetime', inplace=True)
        df.index.name = 'Date'
    elif 'index' in df.columns:
        df['index'] = pd.to_datetime(df['index'])
        if hasattr(df['index'].dt, 'tz') and df['index'].dt.tz is not None:
            df['index'] = df['index'].dt.tz_localize(None)
        df.set_index('index', inplace=True)
        df.index.name = 'Date'
    else:
        try:
            df.index = pd.to_datetime(df.index)
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            df.index.name = 'Date'
        except Exception:
            pass
    return df

def format_df_for_display(df):
    if df is None or df.empty:
        return df
    display_df = df.copy()
    if isinstance(display_df.index, pd.DatetimeIndex):
        display_df.index = display_df.index.strftime('%Y-%m-%d')
    for col in display_df.columns:
        if pd.api.types.is_numeric_dtype(display_df[col]):
            display_df[col] = display_df[col].apply(lambda x: smart_format(x))
    return display_df

def is_crypto_ticker(ticker):
    if not ticker:
        return False
    t_upper = ticker.upper()
    return (
        t_upper.endswith('-USD') or
        t_upper.endswith('-IDR') or
        'USD' in t_upper or
        t_upper in ['BTC-USD', 'ETH-USD', 'BNB-USD', 'SOL-USD', 'XRP-USD', 'ADA-USD', 'DOGE-USD', 'DOT-USD', 'SHIB-USD', 'AVAX-USD', 'BTC', 'ETH', 'SOL', 'DOGE', 'XRP', 'ADA', 'BNB']
    )

def get_cmc_api_key_from_env_or_secrets():
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        sec = user_secrets.get_secret("CMC_API_KEY")
        if sec:
            return sec
    except Exception:
        pass
    try:
        if "CMC_API_KEY" in st.secrets:
            return st.secrets["CMC_API_KEY"]
    except Exception:
        pass
    import os
    return os.environ.get("CMC_API_KEY", "")

PLOTLY_CHART_CONFIG = {
    'scrollZoom': True,
    'displayModeBar': True,
    'displaylogo': False,
    'modeBarButtonsToAdd': [
        'drawline', 'drawopenpath', 'drawcircle', 'drawrect', 'eraseshape'
    ],
    'toImageButtonOptions': {
        'format': 'png',
        'scale': 2
    }
}

def render_plotly_with_tools(fig, key=None):
    if fig is None:
        return
    st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CHART_CONFIG, key=key)

def format_timestamp_for_plot(dt):
    try:
        dt_p = pd.to_datetime(dt)
        if dt_p.hour != 0 or dt_p.minute != 0 or dt_p.second != 0:
            return dt_p.strftime('%Y-%m-%d %H:%M:%S')
        return dt_p.strftime('%Y-%m-%d')
    except Exception:
        return str(dt)

def get_time_change_pct(df, delta):
    if df is None or df.empty or len(df) < 2:
        return None
    try:
        last_dt = pd.to_datetime(df['Date'].iloc[-1] if 'Date' in df.columns else df.index[-1])
        first_dt = pd.to_datetime(df['Date'].iloc[0] if 'Date' in df.columns else df.index[0])
        total_span_sec = (last_dt - first_dt).total_seconds()
        req_sec = delta.total_seconds()
        
        # Jika durasi data yang tersedia tidak mencapai 70% dari target lookback, jangan tampilkan data (null)
        if total_span_sec < (req_sec * 0.7):
            return None
            
        target_dt = last_dt - delta
        dates = pd.to_datetime(df['Date'] if 'Date' in df.columns else df.index)
        sub = df[dates <= target_dt]
        if not sub.empty:
            past_val = safe_float(sub['Close'].iloc[-1])
        else:
            return None
            
        now_val = safe_float(df['Close'].iloc[-1])
        if past_val > 0:
            return ((now_val - past_val) / past_val) * 100.0
        return None
    except Exception:
        return None

def get_time_period_slice(df, delta):
    if df is None or df.empty or len(df) < 2:
        return pd.DataFrame()
    try:
        last_dt = pd.to_datetime(df['Date'].iloc[-1] if 'Date' in df.columns else df.index[-1])
        first_dt = pd.to_datetime(df['Date'].iloc[0] if 'Date' in df.columns else df.index[0])
        total_span_sec = (last_dt - first_dt).total_seconds()
        req_sec = delta.total_seconds()
        
        # Jika durasi data yang tersedia tidak mencapai 70% dari target lookback, jangan tampilkan grafik mini
        if total_span_sec < (req_sec * 0.7):
            return pd.DataFrame()
            
        target_dt = last_dt - delta
        dates = pd.to_datetime(df['Date'] if 'Date' in df.columns else df.index)
        sub = df[dates >= target_dt].copy()
        if len(sub) >= 2:
            return sub
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=300)
def fetch_coinmarketcap_data(symbol, api_key="", interval="1m", count=1000):
    clean_sym = symbol.replace('-USD', '').replace('-IDR', '').replace(' ', '').upper()
    df = pd.DataFrame()
    
    # 1. Coba request langsung ke CoinMarketCap API jika API key tersedia
    if api_key:
        try:
            import requests
            headers = {
                'X-CMC_PRO_API_KEY': api_key.strip(),
                'Accept': 'application/json'
            }
            url = f"https://pro-api.coinmarketcap.com/v2/cryptocurrency/quotes/historical"
            params = {
                'symbol': clean_sym,
                'interval': interval if interval in ['1m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d'] else '1m',
                'count': min(count, 1000)
            }
            resp = requests.get(url, headers=headers, params=params, timeout=8)
            if resp.status_code == 200:
                res_data = resp.json()
                quotes = res_data.get('data', {}).get('quotes', [])
                if quotes:
                    rows = []
                    for q in quotes:
                        ts = q.get('timestamp')
                        quote_usd = q.get('quote', {}).get('USD', {})
                        rows.append({
                            'Date': pd.to_datetime(ts),
                            'Open': safe_float(quote_usd.get('open', quote_usd.get('price', 0.0))),
                            'High': safe_float(quote_usd.get('high', quote_usd.get('price', 0.0))),
                            'Low': safe_float(quote_usd.get('low', quote_usd.get('price', 0.0))),
                            'Close': safe_float(quote_usd.get('price', 0.0)),
                            'Volume': safe_float(quote_usd.get('volume_24h', quote_usd.get('volume', 0.0)))
                        })
                    df = pd.DataFrame(rows)
                    if not df.empty:
                        df = df.set_index('Date').sort_index()
        except Exception:
            pass

    # 2. Fallback intraday live stream
    if df.empty or len(df) < 30:
        try:
            yf_ticker = f"{clean_sym}-USD"
            yf_interval = interval if interval in ['1m', '5m', '15m', '30m', '1h', '1d'] else '1m'
            if yf_interval == '1m':
                yf_period = '7d'
            elif yf_interval in ['5m', '15m', '30m']:
                yf_period = '30d'
            else:
                yf_period = '60d'
                
            df = yf.download(yf_ticker, period=yf_period, interval=yf_interval)
            if not df.empty:
                df = ensure_datetime_index(df)
        except Exception:
            pass

    if not df.empty:
        df = ensure_datetime_index(df)
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col not in df.columns:
                df[col] = df.iloc[:, 0]
    return df

@st.cache_data(ttl=3600)
def get_market_cap(ticker, last_close=0.0, last_volume=0.0):
    try:
        t = yf.Ticker(ticker if '-USD' in ticker or '.JK' in ticker else f"{ticker}-USD")
        info = t.info or {}
        cap = info.get('marketCap')
        if cap is None or cap == 0:
            cap = info.get('regularMarketVolume')
        if cap is None or cap == 0:
            circ_supply = info.get('circulatingSupply')
            if circ_supply and last_close and last_close > 0:
                cap = float(circ_supply) * float(last_close)
        if (cap is None or cap == 0) and last_volume and last_close and last_volume > 0 and last_close > 0:
            cap = float(last_volume) * float(last_close)
        return cap
    except Exception:
        if last_volume and last_close and last_volume > 0 and last_close > 0:
            return float(last_volume) * float(last_close)
        return None

def format_market_cap(cap, curr_prefix=""):
    if cap is None or cap == 0 or pd.isna(cap):
        return "-"
    try:
        fcap = float(cap)
        if fcap >= 1e12:
            return f"{curr_prefix}{fcap/1e12:,.2f} T"
        elif fcap >= 1e9:
            return f"{curr_prefix}{fcap/1e9:,.2f} B"
        elif fcap >= 1e6:
            return f"{curr_prefix}{fcap/1e6:,.2f} M"
        else:
            return smart_format(fcap, prefix=curr_prefix)
    except Exception:
        return "-"

def get_change_pct(df, days_lookback):
    if df is None or len(df) < 2:
        return None
    try:
        last_dt = pd.to_datetime(df['Date'].iloc[-1] if 'Date' in df.columns else df.index[-1])
        first_dt = pd.to_datetime(df['Date'].iloc[0] if 'Date' in df.columns else df.index[0])
        total_span_days = (last_dt - first_dt).total_seconds() / 86400.0
        
        # Jika durasi data yang tersedia tidak mencapai 70% dari lookback hari, jangan tampilkan (null)
        if days_lookback > 1 and total_span_days < (days_lookback * 0.7):
            return None
            
        target_dt = last_dt - pd.Timedelta(days=days_lookback)
        dates = pd.to_datetime(df['Date'] if 'Date' in df.columns else df.index)
        sub_df = df[dates <= target_dt]
        if not sub_df.empty:
            past_price = safe_float(sub_df['Close'].iloc[-1])
        else:
            return None
        
        curr_price = safe_float(df['Close'].iloc[-1])
        if past_price > 0:
            return ((curr_price - past_price) / past_price) * 100.0
        return None
    except Exception:
        return None

def get_period_slice(df, days):
    if df is None or df.empty or len(df) < 2:
        return pd.DataFrame()
    try:
        last_dt = pd.to_datetime(df['Date'].iloc[-1] if 'Date' in df.columns else df.index[-1])
        first_dt = pd.to_datetime(df['Date'].iloc[0] if 'Date' in df.columns else df.index[0])
        total_span_days = (last_dt - first_dt).total_seconds() / 86400.0
        
        # Jika durasi data tidak mencukupi, jangan tampilkan grafik
        if days > 1 and total_span_days < (days * 0.7):
            return pd.DataFrame()
            
        start_dt = last_dt - pd.Timedelta(days=days)
        dates = pd.to_datetime(df['Date'] if 'Date' in df.columns else df.index)
        sub = df[dates >= start_dt].copy()
        if len(sub) >= 2:
            return sub
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def render_coinmarketcap_mini_metrics(quick_df, symbol):
    if quick_df is None or quick_df.empty or len(quick_df) < 2:
        return
        
    c_now = safe_float(quick_df['Close'].iloc[-1])
    c_prev = safe_float(quick_df['Close'].iloc[-2])
    
    chg_1m = ((c_now - c_prev) / c_prev) * 100.0 if c_prev > 0 else 0.0
    chg_3m = get_time_change_pct(quick_df, timedelta(minutes=3))
    chg_5m = get_time_change_pct(quick_df, timedelta(minutes=5))
    chg_15m = get_time_change_pct(quick_df, timedelta(minutes=15))
    chg_30m = get_time_change_pct(quick_df, timedelta(minutes=30))
    chg_1h = get_time_change_pct(quick_df, timedelta(hours=1))
    chg_2h = get_time_change_pct(quick_df, timedelta(hours=2))
    chg_4h = get_time_change_pct(quick_df, timedelta(hours=4))
    chg_12h = get_time_change_pct(quick_df, timedelta(hours=12))
    chg_1d = get_time_change_pct(quick_df, timedelta(days=1))
    
    st.markdown(f"**Performa Perubahan Harga {symbol} (CoinMarketCap):**")
    
    # Baris 1: Intraday Menit (1m, 3m, 5m, 15m, 30m)
    r1_cols = st.columns(5)
    metrics_r1 = [
        (r1_cols[0], "1 Menit (1m)", chg_1m),
        (r1_cols[1], "3 Menit (3m)", chg_3m),
        (r1_cols[2], "5 Menit (5m)", chg_5m),
        (r1_cols[3], "15 Menit (15m)", chg_15m),
        (r1_cols[4], "30 Menit (30m)", chg_30m)
    ]
    for col, label, val in metrics_r1:
        with col:
            if val is not None and not pd.isna(val):
                color = "#00C853" if val >= 0 else "#D50000"
                symbol_arrow = "▲" if val >= 0 else "▼"
                sign = "+" if val > 0 else ""
                bg_color = "rgba(0, 200, 83, 0.08)" if val >= 0 else "rgba(213, 0, 0, 0.08)"
                st.markdown(f"""
                <div style="background-color: {bg_color}; padding: 8px 6px; border-radius: 8px; border-left: 4px solid {color}; text-align: center; margin-bottom: 6px;">
                    <div style="font-size: 11px; color: #555; font-weight: 600;">{label}</div>
                    <div style="font-size: 14px; font-weight: bold; color: {color}; margin-top: 2px;">{symbol_arrow} {sign}{val:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background-color: #f5f5f5; padding: 8px 6px; border-radius: 8px; text-align: center; margin-bottom: 6px;">
                    <div style="font-size: 11px; color: #777; font-weight: 600;">{label}</div>
                    <div style="font-size: 14px; font-weight: bold; color: #999; margin-top: 2px;">-</div>
                </div>
                """, unsafe_allow_html=True)
                
    # Baris 2: Intraday Jam & Harian (1H, 2H, 4H, 12H, 1D)
    r2_cols = st.columns(5)
    metrics_r2 = [
        (r2_cols[0], "1 Jam (1H)", chg_1h),
        (r2_cols[1], "2 Jam (2H)", chg_2h),
        (r2_cols[2], "4 Jam (4H)", chg_4h),
        (r2_cols[3], "12 Jam (12H)", chg_12h),
        (r2_cols[4], "1 Hari (1D)", chg_1d)
    ]
    for col, label, val in metrics_r2:
        with col:
            if val is not None and not pd.isna(val):
                color = "#00C853" if val >= 0 else "#D50000"
                symbol_arrow = "▲" if val >= 0 else "▼"
                sign = "+" if val > 0 else ""
                bg_color = "rgba(0, 200, 83, 0.08)" if val >= 0 else "rgba(213, 0, 0, 0.08)"
                st.markdown(f"""
                <div style="background-color: {bg_color}; padding: 8px 6px; border-radius: 8px; border-left: 4px solid {color}; text-align: center;">
                    <div style="font-size: 11px; color: #555; font-weight: 600;">{label}</div>
                    <div style="font-size: 14px; font-weight: bold; color: {color}; margin-top: 2px;">{symbol_arrow} {sign}{val:.2f}%</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background-color: #f5f5f5; padding: 8px 6px; border-radius: 8px; text-align: center;">
                    <div style="font-size: 11px; color: #777; font-weight: 600;">{label}</div>
                    <div style="font-size: 14px; font-weight: bold; color: #999; margin-top: 2px;">-</div>
                </div>
                """, unsafe_allow_html=True)

def extract_1d_array(data):
    if data is None:
        return np.array([], dtype=float)
    if isinstance(data, pd.DataFrame):
        data = data.iloc[:, 0]
    if isinstance(data, pd.Series):
        arr = pd.to_numeric(data, errors='coerce').dropna().values
    else:
        arr = np.asarray(data, dtype=float)
    arr = arr.squeeze().ravel()
    return arr[~np.isnan(arr)]

def calculate_vwap_series(df_slice):
    if df_slice is None or df_slice.empty:
        return np.array([], dtype=float)
    try:
        close_arr = extract_1d_array(df_slice['Close'] if 'Close' in df_slice.columns else df_slice.iloc[:, 0])
        if len(close_arr) == 0:
            return np.array([], dtype=float)
        
        high_arr = extract_1d_array(df_slice['High']) if 'High' in df_slice.columns else close_arr
        low_arr = extract_1d_array(df_slice['Low']) if 'Low' in df_slice.columns else close_arr
        vol_arr = extract_1d_array(df_slice['Volume']) if 'Volume' in df_slice.columns else np.ones_like(close_arr)

        min_len = min(len(close_arr), len(high_arr), len(low_arr), len(vol_arr))
        if min_len == 0:
            return close_arr
        c = close_arr[:min_len]
        h = high_arr[:min_len]
        l = low_arr[:min_len]
        v = vol_arr[:min_len]

        tp = (h + l + c) / 3.0
        cum_tp_vol = np.cumsum(tp * v)
        cum_vol = np.cumsum(v)
        cum_vol[cum_vol == 0] = np.nan
        vwap = cum_tp_vol / cum_vol
        vwap[np.isnan(vwap)] = c[np.isnan(vwap)]
        return vwap
    except Exception:
        return extract_1d_array(df_slice['Close'] if 'Close' in df_slice.columns else df_slice.iloc[:, 0])

def calculate_atr_series(df_slice, period=14):
    if df_slice is None or df_slice.empty:
        return np.array([], dtype=float)
    try:
        close_arr = extract_1d_array(df_slice['Close'] if 'Close' in df_slice.columns else df_slice.iloc[:, 0])
        n = len(close_arr)
        if n == 0:
            return np.array([], dtype=float)
        high_arr = extract_1d_array(df_slice['High']) if 'High' in df_slice.columns else close_arr
        low_arr = extract_1d_array(df_slice['Low']) if 'Low' in df_slice.columns else close_arr
        
        min_len = min(len(close_arr), len(high_arr), len(low_arr))
        c = close_arr[:min_len]
        h = high_arr[:min_len]
        l = low_arr[:min_len]
        
        if min_len < 2:
            return np.array([max(0.0, h[0] - l[0])], dtype=float)
            
        tr = np.zeros(min_len)
        tr[0] = h[0] - l[0]
        for i in range(1, min_len):
            tr[i] = max(h[i] - l[i], abs(h[i] - c[i-1]), abs(l[i] - c[i-1]))
            
        atr = pd.Series(tr).ewm(span=min(period, max(2, min_len)), adjust=False).mean().values
        return atr
    except Exception:
        return np.array([], dtype=float)

def calculate_daily_delta_volume_series(df_slice):
    if df_slice is None or df_slice.empty:
        return np.array([], dtype=float), []
    try:
        close_arr = extract_1d_array(df_slice['Close'] if 'Close' in df_slice.columns else df_slice.iloc[:, 0])
        n = len(close_arr)
        if n == 0:
            return np.array([], dtype=float), []
        open_arr = extract_1d_array(df_slice['Open']) if 'Open' in df_slice.columns else None
        vol_arr = extract_1d_array(df_slice['Volume']) if 'Volume' in df_slice.columns else np.ones(n)
        
        min_len = min(len(close_arr), len(vol_arr))
        c = close_arr[:min_len]
        v = vol_arr[:min_len]
        
        delta_vol = np.zeros(min_len)
        bar_colors = []
        for i in range(min_len):
            if open_arr is not None and i < len(open_arr) and not np.isnan(open_arr[i]):
                is_up = c[i] >= open_arr[i]
            elif i > 0:
                is_up = c[i] >= c[i-1]
            else:
                is_up = True
            delta_vol[i] = v[i] if is_up else -v[i]
            bar_colors.append('#00C853' if is_up else '#D50000')
            
        return delta_vol, bar_colors
    except Exception:
        return np.array([], dtype=float), []

def calculate_delta_volume_series(df_slice):
    delta_vol, _ = calculate_daily_delta_volume_series(df_slice)
    return np.cumsum(delta_vol) if len(delta_vol) > 0 else np.array([], dtype=float)

def get_bar_colors_for_volume(df_slice):
    if df_slice is None or df_slice.empty:
        return None
    try:
        close_arr = extract_1d_array(df_slice['Close'] if 'Close' in df_slice.columns else df_slice.iloc[:, 0])
        open_arr = extract_1d_array(df_slice['Open']) if 'Open' in df_slice.columns else None
        
        n = len(close_arr)
        if n == 0:
            return None
            
        colors = []
        for i in range(n):
            if open_arr is not None and i < len(open_arr) and not np.isnan(open_arr[i]):
                is_up = close_arr[i] >= open_arr[i]
            elif i > 0:
                is_up = close_arr[i] >= close_arr[i-1]
            else:
                is_up = True
            colors.append('#00C853' if is_up else '#D50000')
        return colors
    except Exception:
        return None

def render_combined_volume_atr_sparkline(vol_series, atr_series, bar_colors=None, height=1.3):
    fig, ax1 = plt.subplots(figsize=(2.5, height), dpi=100)
    fig.patch.set_facecolor('none')
    ax1.set_facecolor('none')
    
    y_vol = extract_1d_array(vol_series)
    y_atr = extract_1d_array(atr_series)
    
    if len(y_vol) == 0:
        ax1.text(0.5, 0.5, '-', ha='center', va='center', fontsize=9, color='#888')
    elif len(y_vol) == 1:
        c = bar_colors[0] if (bar_colors is not None and len(bar_colors) > 0) else '#00C853'
        ax1.bar([0], [y_vol[0]], color=c, alpha=0.85, width=0.6)
        if len(y_atr) > 0:
            ax2 = ax1.twinx()
            ax2.set_facecolor('none')
            ax2.scatter([0], [y_atr[0]], color='#FFB300', s=20)
            for spine in ax2.spines.values():
                spine.set_visible(False)
            ax2.set_xticks([])
            ax2.set_yticks([])
    else:
        x = np.arange(len(y_vol), dtype=float)
        if bar_colors is not None and len(bar_colors) == len(y_vol):
            ax1.bar(x, y_vol, color=bar_colors, alpha=0.75, width=0.8)
        else:
            ax1.bar(x, y_vol, color='#00C853', alpha=0.75, width=0.8)
            
        if len(y_atr) == len(y_vol) and len(y_atr) > 1:
            ax2 = ax1.twinx()
            ax2.set_facecolor('none')
            ax2.plot(x, y_atr, color='#FFB300', linewidth=2.0)
            for spine in ax2.spines.values():
                spine.set_visible(False)
            ax2.set_xticks([])
            ax2.set_yticks([])
            
    for spine in ax1.spines.values():
        spine.set_visible(False)
    ax1.set_xticks([])
    ax1.set_yticks([])
    plt.tight_layout(pad=0.1)
    return fig

def render_combined_price_vwap_sparkline(close_series, vwap_series, is_positive=True, height=1.3):
    fig, ax = plt.subplots(figsize=(2.5, height), dpi=100)
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')
    
    color_close = '#00C853' if is_positive else '#D50000'
    color_vwap = '#00B0FF' # Warna biru khas untuk VWAP
    
    y_close = extract_1d_array(close_series)
    y_vwap = extract_1d_array(vwap_series)
    
    if len(y_close) == 0:
        ax.text(0.5, 0.5, '-', ha='center', va='center', fontsize=9, color='#888')
    elif len(y_close) == 1:
        ax.scatter([0], [y_close[0]], color=color_close, s=22)
        if len(y_vwap) > 0:
            ax.scatter([0], [y_vwap[0]], color=color_vwap, s=16)
    else:
        x = np.arange(len(y_close), dtype=float)
        # Plot Close Price with subtle gradient fill
        ax.plot(x, y_close, color=color_close, linewidth=2.0)
        min_y = float(np.min(y_close))
        try:
            ax.fill_between(x, y_close, min_y, color=color_close, alpha=0.12)
        except Exception:
            pass
            
        # Plot VWAP line (Blue dashed)
        if len(y_vwap) == len(y_close):
            ax.plot(x, y_vwap, color=color_vwap, linewidth=1.7, linestyle='--')
            
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout(pad=0.1)
    return fig

def render_sparkline_chart(series, is_positive=True, chart_type='line', fill=True, height=1.3, bar_colors=None, line_color=None):
    fig, ax = plt.subplots(figsize=(2.5, height), dpi=100)
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')
    
    if line_color is not None:
        color = line_color
    else:
        color = '#00C853' if is_positive else '#D50000'
    
    y = extract_1d_array(series)

    if len(y) == 0:
        ax.text(0.5, 0.5, '-', ha='center', va='center', fontsize=9, color='#888')
    elif len(y) == 1:
        if chart_type == 'bar':
            c = bar_colors[0] if (bar_colors is not None and len(bar_colors) > 0) else color
            ax.bar([0], [y[0]], color=c, alpha=0.85, width=0.6)
        else:
            ax.scatter([0], [y[0]], color=color, s=20)
    else:
        x = np.arange(len(y), dtype=float)
        if chart_type == 'bar':
            if bar_colors is not None and len(bar_colors) == len(y):
                ax.bar(x, y, color=bar_colors, alpha=0.85, width=0.8)
            else:
                ax.bar(x, y, color=color, alpha=0.85, width=0.8)
        else:
            ax.plot(x, y, color=color, linewidth=2.0)
            if fill:
                min_y = float(np.min(y))
                try:
                    ax.fill_between(x, y, min_y, color=color, alpha=0.15)
                except Exception:
                    pass
                
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout(pad=0.1)
    return fig

def plot_interactive_history(df, title, y_label, line_color, curr_prefix=""):
    fig = go.Figure()
    if df is None or df.empty:
        return fig
    
    dates = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df['Date'] if 'Date' in df.columns else df.index)
    close_vals = extract_1d_array(df['Close'] if 'Close' in df.columns else df.iloc[:, 0])
    
    n = min(len(dates), len(close_vals))
    dates = dates[:n]
    close_vals = close_vals[:n]
    
    has_ohlc = all(c in df.columns for c in ['Open', 'High', 'Low', 'Close'])
    if has_ohlc:
        open_vals = extract_1d_array(df['Open'])[:n]
        high_vals = extract_1d_array(df['High'])[:n]
        low_vals = extract_1d_array(df['Low'])[:n]
        vol_vals = extract_1d_array(df['Volume'])[:n] if 'Volume' in df.columns else None
        
        hover_text = []
        for i in range(n):
            d_str = dates[i].strftime('%Y-%m-%d')
            c_str = smart_format(close_vals[i], prefix=curr_prefix)
            o_str = smart_format(open_vals[i], prefix=curr_prefix) if i < len(open_vals) else "-"
            h_str = smart_format(high_vals[i], prefix=curr_prefix) if i < len(high_vals) else "-"
            l_str = smart_format(low_vals[i], prefix=curr_prefix) if i < len(low_vals) else "-"
            v_str = f"{vol_vals[i]:,.0f}" if (vol_vals is not None and i < len(vol_vals)) else "-"
            hover_text.append(
                f"<b>Tanggal:</b> {d_str}<br>"
                f"<b>Close:</b> {c_str}<br>"
                f"<b>Open:</b> {o_str}<br>"
                f"<b>High:</b> {h_str}<br>"
                f"<b>Low:</b> {l_str}<br>"
                f"<b>Volume:</b> {v_str}"
            )
    else:
        hover_text = [
            f"<b>Tanggal:</b> {dates[i].strftime('%Y-%m-%d')}<br><b>{y_label}:</b> {smart_format(close_vals[i], prefix=curr_prefix)}"
            for i in range(n)
        ]

    # Main Price Line
    fig.add_trace(go.Scatter(
        x=dates,
        y=close_vals,
        mode='lines',
        name=y_label,
        line=dict(color=line_color, width=2.2),
        hoverinfo='text',
        hovertext=hover_text
    ))

    # High and Low markers
    if len(close_vals) > 0:
        max_idx = int(np.argmax(close_vals))
        min_idx = int(np.argmin(close_vals))

        max_date, max_val = dates[max_idx], close_vals[max_idx]
        min_date, min_val = dates[min_idx], close_vals[min_idx]

        # High Marker
        fig.add_trace(go.Scatter(
            x=[max_date],
            y=[max_val],
            mode='markers+text',
            name='Tertinggi (High)',
            marker=dict(color='#00C853', size=11, symbol='triangle-up'),
            text=[f"▲ High: {smart_format(max_val, prefix=curr_prefix)}"],
            textposition="top center",
            textfont=dict(color='#00C853', size=12),
            hoverinfo='text',
            hovertext=[f"<b>Tertinggi (High)</b><br>Tanggal: {max_date.strftime('%Y-%m-%d')}<br>Harga: {smart_format(max_val, prefix=curr_prefix)}"]
        ))

        # Low Marker
        fig.add_trace(go.Scatter(
            x=[min_date],
            y=[min_val],
            mode='markers+text',
            name='Terendah (Low)',
            marker=dict(color='#D50000', size=11, symbol='triangle-down'),
            text=[f"▼ Low: {smart_format(min_val, prefix=curr_prefix)}"],
            textposition="bottom center",
            textfont=dict(color='#D50000', size=12),
            hoverinfo='text',
            hovertext=[f"<b>Terendah (Low)</b><br>Tanggal: {min_date.strftime('%Y-%m-%d')}<br>Harga: {smart_format(min_val, prefix=curr_prefix)}"]
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color='#222')),
        xaxis=dict(title="Tanggal", showgrid=True, gridcolor='#F0F0F0'),
        yaxis=dict(title=y_label, showgrid=True, gridcolor='#F0F0F0'),
        hovermode='x unified',
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template='plotly_white',
        height=460
    )
    return fig

def plot_interactive_evaluation(actual_dates, y_test, y_pred, y_label, curr_prefix=""):
    fig = go.Figure()
    n = min(len(actual_dates), len(y_test), len(y_pred))
    dates = pd.to_datetime(actual_dates[:n])
    y_t = extract_1d_array(y_test)[:n]
    y_p = extract_1d_array(y_pred)[:n]

    max_idx = int(np.argmax(y_t)) if n > 0 else -1
    min_idx = int(np.argmin(y_t)) if n > 0 else -1
    base_act_eval = y_t[0] if (n > 0 and y_t[0] > 0) else 1.0
    base_pred_eval = y_p[0] if (n > 0 and y_p[0] > 0) else 1.0

    hover_actual = []
    for i in range(n):
        d_str = dates[i].strftime('%Y-%m-%d')
        p_str = smart_format(y_t[i], prefix=curr_prefix)
        ret_val = y_t[i] - base_act_eval
        ret_pct = (ret_val / base_act_eval) * 100.0 if base_act_eval > 0 else 0.0
        ret_sign = "+" if ret_val > 0 else ("-" if ret_val < 0 else "")
        ret_arrow = "▲" if ret_val >= 0 else "▼"
        ret_color = "#00C853" if ret_val >= 0 else "#D50000"
        ret_diff_str = smart_format(abs(ret_val), prefix=curr_prefix)

        extra_tag = ""
        if i == max_idx:
            extra_tag += '<br><b style="color:#00C853;">▲ [Aktual Tertinggi (High)]</b>'
        if i == min_idx:
            extra_tag += '<br><b style="color:#D50000;">▼ [Aktual Terendah (Low)]</b>'

        hover_actual.append(
            f"<b>Tanggal:</b> {d_str}<br>"
            f"<b>Harga Aktual:</b> {p_str}<br>"
            f"<b>Perubahan dari Basis:</b> <span style='color:{ret_color};'>{ret_arrow} {ret_sign}{abs(ret_pct):.2f}% ({ret_sign}{ret_diff_str})</span>"
            f"{extra_tag}"
        )

    hover_pred = []
    for i in range(n):
        d_str = dates[i].strftime('%Y-%m-%d')
        p_str = smart_format(y_p[i], prefix=curr_prefix)
        ret_val = y_p[i] - base_pred_eval
        ret_pct = (ret_val / base_pred_eval) * 100.0 if base_pred_eval > 0 else 0.0
        ret_sign = "+" if ret_val > 0 else ("-" if ret_val < 0 else "")
        ret_arrow = "▲" if ret_val >= 0 else "▼"
        ret_color = "#00C853" if ret_val >= 0 else "#D50000"
        ret_diff_str = smart_format(abs(ret_val), prefix=curr_prefix)
        diff_val = y_p[i] - y_t[i]
        diff_str = smart_format(diff_val, prefix=curr_prefix)

        hover_pred.append(
            f"<b>Tanggal:</b> {d_str}<br>"
            f"<b>Harga Pengujian (Uji):</b> {p_str}<br>"
            f"<b>Perubahan dari Basis:</b> <span style='color:{ret_color};'>{ret_arrow} {ret_sign}{abs(ret_pct):.2f}% ({ret_sign}{ret_diff_str})</span><br>"
            f"<b>Selisih (Diff vs Aktual):</b> {diff_str}"
        )

    fig.add_trace(go.Scatter(
        x=dates,
        y=y_t,
        mode='lines',
        name='Harga Aktual',
        line=dict(color='#D6C36B', width=2.2),
        hoverinfo='text',
        hovertext=hover_actual
    ))

    fig.add_trace(go.Scatter(
        x=dates,
        y=y_p,
        mode='lines',
        name='Harga Pengujian (Prediksi)',
        line=dict(color='#B16ED0', width=2.2),
        hoverinfo='text',
        hovertext=hover_pred
    ))

    if len(y_t) > 0 and max_idx >= 0 and min_idx >= 0:
        fig.add_trace(go.Scatter(
            x=[dates[max_idx]],
            y=[y_t[max_idx]],
            mode='markers+text',
            name='Aktual Tertinggi',
            marker=dict(color='#00C853', size=11, symbol='triangle-up'),
            text=[f"▲ High: {smart_format(y_t[max_idx], prefix=curr_prefix)}"],
            textposition="top center",
            textfont=dict(color='#00C853', size=12),
            hoverinfo='skip',
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=[dates[min_idx]],
            y=[y_t[min_idx]],
            mode='markers+text',
            name='Aktual Terendah',
            marker=dict(color='#D50000', size=11, symbol='triangle-down'),
            text=[f"▼ Low: {smart_format(y_t[min_idx], prefix=curr_prefix)}"],
            textposition="bottom center",
            textfont=dict(color='#D50000', size=12),
            hoverinfo='skip',
            showlegend=False
        ))

    fig.update_layout(
        title=dict(text='Perbandingan Harga Aktual dan Prediksi', font=dict(size=16, color='#222')),
        xaxis=dict(title="Tanggal", showgrid=True, gridcolor='#F0F0F0'),
        yaxis=dict(title=y_label, showgrid=True, gridcolor='#F0F0F0'),
        hovermode='x unified',
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template='plotly_white',
        height=460
    )
    return fig

def plot_interactive_forecast(hist_dates, hist_prices, actual_dates, y_pred, date_range, forecast, y_label, curr_prefix=""):
    fig = go.Figure()
    today_dt = pd.to_datetime(date.today())
    if hasattr(today_dt, 'tz') and today_dt.tz is not None:
        today_dt = today_dt.tz_localize(None)
    
    h_dates = pd.to_datetime(hist_dates)
    if hasattr(h_dates, 'tz') and h_dates.tz is not None:
        h_dates = h_dates.tz_localize(None)
    h_prices = extract_1d_array(hist_prices)
    n_h = min(len(h_dates), len(h_prices))
    h_dates = h_dates[:n_h]
    h_prices = h_prices[:n_h]
    
    # 1. Hitung Extremum pada Harga Aktual
    act_max_val = np.max(h_prices) if n_h > 0 else -1e9
    act_min_val = np.min(h_prices) if n_h > 0 else 1e9
    act_max_idx = int(np.argmax(h_prices)) if n_h > 0 else -1
    act_min_idx = int(np.argmin(h_prices)) if n_h > 0 else -1
    act_max_dt = h_dates[act_max_idx] if act_max_idx >= 0 else None
    act_min_dt = h_dates[act_min_idx] if act_min_idx >= 0 else None

    # Base price untuk return aktual (titik awal slice grafik)
    base_act_pr = h_prices[0] if n_h > 0 else 1.0

    # Hover text untuk Harga Aktual
    hover_actual = []
    for i in range(n_h):
        d_str = format_timestamp_for_plot(h_dates[i])
        p_str = smart_format(h_prices[i], prefix=curr_prefix)
        ret_val = h_prices[i] - base_act_pr
        ret_pct = (ret_val / base_act_pr) * 100.0 if base_act_pr > 0 else 0.0
        ret_sign = "+" if ret_val > 0 else ("-" if ret_val < 0 else "")
        ret_arrow = "▲" if ret_val >= 0 else "▼"
        ret_color = "#00C853" if ret_val >= 0 else "#D50000"
        ret_diff_str = smart_format(abs(ret_val), prefix=curr_prefix)
        
        extra_tag = ""
        if i == act_max_idx:
            extra_tag += '<br><b style="color:#00C853;">▲ [Tertinggi (High Aktual)]</b>'
        if i == act_min_idx:
            extra_tag += '<br><b style="color:#D50000;">▼ [Terendah (Low Aktual)]</b>'

        hover_actual.append(
            f"<b>Tanggal:</b> {d_str}<br>"
            f"<b>Harga Aktual:</b> {p_str}<br>"
            f"<b>Perubahan dari Basis:</b> <span style='color:{ret_color};'>{ret_arrow} {ret_sign}{abs(ret_pct):.2f}% ({ret_sign}{ret_diff_str})</span>"
            f"{extra_tag}"
        )

    # Trace 1: Garis Harga Aktual
    fig.add_trace(go.Scatter(
        x=h_dates,
        y=h_prices,
        mode='lines',
        name='Harga Aktual',
        line=dict(color='#D6C36B', width=2.2),
        hoverinfo='text',
        hovertext=hover_actual
    ))

    # Trace 2: Harga Pengujian (Uji)
    last_test_date = None
    last_test_price = None
    if actual_dates is not None and y_pred is not None and len(actual_dates) > 0:
        a_dates = pd.to_datetime(actual_dates)
        if hasattr(a_dates, 'tz') and a_dates.tz is not None:
            a_dates = a_dates.tz_localize(None)
        p_prices = extract_1d_array(y_pred)
        n_a = min(len(a_dates), len(p_prices))
        if n_a > 0:
            a_dates = a_dates[:n_a]
            p_prices = p_prices[:n_a]
            last_test_date = a_dates[-1]
            last_test_price = p_prices[-1]
            base_test_pr = p_prices[0] if p_prices[0] > 0 else 1.0
            
            hover_test = []
            for i in range(n_a):
                d_str = format_timestamp_for_plot(a_dates[i])
                p_str = smart_format(p_prices[i], prefix=curr_prefix)
                ret_val = p_prices[i] - base_test_pr
                ret_pct = (ret_val / base_test_pr) * 100.0 if base_test_pr > 0 else 0.0
                ret_sign = "+" if ret_val > 0 else ("-" if ret_val < 0 else "")
                ret_arrow = "▲" if ret_val >= 0 else "▼"
                ret_color = "#00C853" if ret_val >= 0 else "#D50000"
                ret_diff_str = smart_format(abs(ret_val), prefix=curr_prefix)
                
                hover_test.append(
                    f"<b>Tanggal:</b> {d_str}<br>"
                    f"<b>Harga Pengujian (Uji):</b> {p_str}<br>"
                    f"<b>Perubahan dari Basis:</b> <span style='color:{ret_color};'>{ret_arrow} {ret_sign}{abs(ret_pct):.2f}% ({ret_sign}{ret_diff_str})</span>"
                )

            fig.add_trace(go.Scatter(
                x=a_dates,
                y=p_prices,
                mode='lines',
                name='Harga Pengujian (Uji)',
                line=dict(color='#B16ED0', width=2.2),
                hoverinfo='text',
                hovertext=hover_test
            ))

    f_dates = pd.to_datetime(date_range)
    if hasattr(f_dates, 'tz') and f_dates.tz is not None:
        f_dates = f_dates.tz_localize(None)
    f_prices = extract_1d_array(forecast)
    n_f = min(len(f_dates), len(f_prices))
    f_dates = f_dates[:n_f]
    f_prices = f_prices[:n_f]

    proj_prices = np.array([])
    proj_max_dt = None
    proj_min_dt = None
    proj_max_val = -1e9
    proj_min_val = 1e9

    if n_f > 0:
        last_act_dt = h_dates[-1]
        last_act_pr = h_prices[-1]
        base_ref = last_test_price if (last_test_price is not None and last_test_price > 0) else f_prices[0]

        # Hitung Proyeksi Tren Aktual (nominal riil)
        proj_list = []
        for p in f_prices:
            pct_rel = (p - base_ref) / base_ref if base_ref > 0 else 0.0
            proj_val = last_act_pr * (1.0 + pct_rel)
            proj_list.append(proj_val)
        proj_prices = np.array(proj_list)

        proj_max_val = np.max(proj_prices)
        proj_min_val = np.min(proj_prices)
        proj_max_idx = int(np.argmax(proj_prices))
        proj_min_idx = int(np.argmin(proj_prices))
        proj_max_dt = f_dates[proj_max_idx]
        proj_min_dt = f_dates[proj_min_idx]

        # =========================================================================
        # --- A. HARGA PREDIKSI MODEL (Ditaruh DULUAN agar posisinya di background) ---
        # =========================================================================
        # 1. Transisi dari Uji Terakhir ke Prediksi Pertama
        if last_test_date is not None and last_test_price is not None:
            is_past_or_today_p0 = (f_dates[0] <= today_dt)
            dash_p0 = 'dot' if is_past_or_today_p0 else 'solid'
            fig.add_trace(go.Scatter(
                x=[last_test_date, f_dates[0]],
                y=[last_test_price, f_prices[0]],
                mode='lines',
                line=dict(color='#107EDE', width=2.0, dash=dash_p0),
                showlegend=False,
                hoverinfo='skip'
            ))

        # 2. Segmen Prediksi Model berikutnya
        for idx in range(n_f - 1):
            d_next = f_dates[idx + 1]
            dash_p = 'dot' if (d_next <= today_dt) else 'solid'
            fig.add_trace(go.Scatter(
                x=[f_dates[idx], d_next],
                y=[f_prices[idx], f_prices[idx + 1]],
                mode='lines+markers',
                line=dict(color='#107EDE', width=2.0, dash=dash_p),
                marker=dict(size=5, color='#107EDE'),
                showlegend=False,
                hoverinfo='skip'
            ))

        # 3. Master Hover Trace untuk Harga Prediksi Model
        hover_pred_fc = []
        base_test_ref = last_test_price if (last_test_price is not None and last_test_price > 0) else f_prices[0]
        for i in range(n_f):
            d_str = format_timestamp_for_plot(f_dates[i])
            p_str = smart_format(f_prices[i], prefix=curr_prefix)
            ret_val = f_prices[i] - base_test_ref
            ret_pct = (ret_val / base_test_ref) * 100.0 if base_test_ref > 0 else 0.0
            ret_sign = "+" if ret_val > 0 else ("-" if ret_val < 0 else "")
            ret_arrow = "▲" if ret_val >= 0 else "▼"
            ret_color = "#00C853" if ret_val >= 0 else "#D50000"
            ret_diff_str = smart_format(abs(ret_val), prefix=curr_prefix)

            hover_pred_fc.append(
                f"<b>Tanggal:</b> {d_str}<br>"
                f"<b>Harga Prediksi Model:</b> {p_str}<br>"
                f"<b>Estimasi Return Model:</b> <span style='color:{ret_color};'>{ret_arrow} {ret_sign}{abs(ret_pct):.2f}% ({ret_sign}{ret_diff_str})</span>"
            )

        fig.add_trace(go.Scatter(
            x=f_dates,
            y=f_prices,
            mode='markers',
            name='Harga Prediksi Model',
            marker=dict(size=5, color='#107EDE'),
            hoverinfo='text',
            hovertext=hover_pred_fc,
            showlegend=True
        ))

        # =========================================================================
        # --- B. PROYEKSI TREN AKTUAL (Ditaruh SETELAHNYA agar berada di FOREGROUND / DEPAN) ---
        # =========================================================================
        # 1. Transisi dari Aktual Terakhir ke Titik Proyeksi Pertama
        is_past_or_today_0 = (f_dates[0] <= today_dt)
        dash_style_0 = 'dot' if is_past_or_today_0 else 'solid'
        up_0 = (proj_prices[0] >= last_act_pr)
        col_0 = '#00C853' if up_0 else '#D50000'

        fig.add_trace(go.Scatter(
            x=[last_act_dt, f_dates[0]],
            y=[last_act_pr, proj_prices[0]],
            mode='lines+markers',
            line=dict(color=col_0, width=2.6, dash=dash_style_0),
            marker=dict(size=8, color=col_0),
            showlegend=False,
            hoverinfo='skip'
        ))

        # 2. Segmen-segmen Proyeksi berikutnya (f_dates[idx] -> f_dates[idx+1])
        for idx in range(n_f - 1):
            d_next = f_dates[idx + 1]
            p_curr = proj_prices[idx]
            p_next = proj_prices[idx + 1]
            
            is_past_or_today = (d_next <= today_dt)
            dash_style = 'dot' if is_past_or_today else 'solid'
            seg_up = (p_next >= p_curr)
            seg_col = '#00C853' if seg_up else '#D50000'

            fig.add_trace(go.Scatter(
                x=[f_dates[idx], d_next],
                y=[p_curr, p_next],
                mode='lines+markers',
                line=dict(color=seg_col, width=2.6, dash=dash_style),
                marker=dict(size=8, color=seg_col),
                showlegend=False,
                hoverinfo='skip'
            ))

        # 3. Hover Unifikasi untuk Proyeksi Tren Aktual
        hover_proj = []
        for i in range(n_f):
            d_str = format_timestamp_for_plot(f_dates[i])
            pr_str = smart_format(proj_prices[i], prefix=curr_prefix)
            ret_val = proj_prices[i] - last_act_pr
            ret_pct = (ret_val / last_act_pr) * 100.0 if last_act_pr > 0 else 0.0
            ret_sign = "+" if ret_val > 0 else ("-" if ret_val < 0 else "")
            ret_arrow = "▲" if ret_val >= 0 else "▼"
            ret_color = "#00C853" if ret_val >= 0 else "#D50000"
            ret_diff_str = smart_format(abs(ret_val), prefix=curr_prefix)

            extra_tag = ""
            if i == proj_max_idx and proj_max_dt != act_max_dt:
                extra_tag += '<br><b style="color:#00C853;">▲ [Tertinggi (High Proyeksi)]</b>'
            if i == proj_min_idx and proj_min_dt != act_min_dt:
                extra_tag += '<br><b style="color:#D50000;">▼ [Terendah (Low Proyeksi)]</b>'

            hover_proj.append(
                f"<b>Tanggal:</b> {d_str}<br>"
                f"<b>Proyeksi Tren Aktual:</b> {pr_str}<br>"
                f"<b>Estimasi Return:</b> <span style='color:{ret_color};'>{ret_arrow} {ret_sign}{abs(ret_pct):.2f}% ({ret_sign}{ret_diff_str})</span>"
                f"{extra_tag}"
            )

        # Single master trace for Proyeksi Tren Aktual (untuk Legend & Hover)
        marker_colors = ['#00C853' if (proj_prices[i] >= (proj_prices[i-1] if i > 0 else last_act_pr)) else '#D50000' for i in range(n_f)]
        fig.add_trace(go.Scatter(
            x=f_dates,
            y=proj_prices,
            mode='markers',
            name='Proyeksi Tren Aktual (🟢 Naik / 🔴 Turun)',
            marker=dict(size=8, color=marker_colors),
            hoverinfo='text',
            hovertext=hover_proj,
            showlegend=True
        ))

    # --- C. PENANDA HIGH & LOW PADA CHART (Deduplicated Text Markers) ---
    # 1. High Aktual
    if act_max_dt is not None:
        fig.add_trace(go.Scatter(
            x=[act_max_dt],
            y=[act_max_val],
            mode='markers+text',
            name='Tertinggi (High Aktual)',
            marker=dict(color='#00C853', size=11, symbol='triangle-up'),
            text=[f"▲ High: {smart_format(act_max_val, prefix=curr_prefix)}"],
            textposition="top center",
            textfont=dict(color='#00C853', size=11),
            hoverinfo='skip',
            showlegend=False
        ))

    # 2. High Proyeksi (hanya jika tanggalnya berbeda dengan High Aktual dan tanggal aktual terakhir)
    if proj_max_dt is not None and proj_max_dt != act_max_dt and (n_h > 0 and proj_max_dt != h_dates[-1]):
        fig.add_trace(go.Scatter(
            x=[proj_max_dt],
            y=[proj_max_val],
            mode='markers+text',
            name='Tertinggi (High Proyeksi)',
            marker=dict(color='#00C853', size=11, symbol='triangle-up'),
            text=[f"▲ High: {smart_format(proj_max_val, prefix=curr_prefix)}"],
            textposition="top center",
            textfont=dict(color='#00C853', size=11),
            hoverinfo='skip',
            showlegend=False
        ))

    # 3. Low Aktual
    if act_min_dt is not None:
        fig.add_trace(go.Scatter(
            x=[act_min_dt],
            y=[act_min_val],
            mode='markers+text',
            name='Terendah (Low Aktual)',
            marker=dict(color='#D50000', size=11, symbol='triangle-down'),
            text=[f"▼ Low: {smart_format(act_min_val, prefix=curr_prefix)}"],
            textposition="bottom center",
            textfont=dict(color='#D50000', size=11),
            hoverinfo='skip',
            showlegend=False
        ))

    # 4. Low Proyeksi (hanya jika tanggalnya berbeda dengan Low Aktual dan tanggal aktual terakhir)
    if proj_min_dt is not None and proj_min_dt != act_min_dt and (n_h > 0 and proj_min_dt != h_dates[-1]):
        fig.add_trace(go.Scatter(
            x=[proj_min_dt],
            y=[proj_min_val],
            mode='markers+text',
            name='Terendah (Low Proyeksi)',
            marker=dict(color='#D50000', size=11, symbol='triangle-down'),
            text=[f"▼ Low: {smart_format(proj_min_val, prefix=curr_prefix)}"],
            textposition="bottom center",
            textfont=dict(color='#D50000', size=11),
            hoverinfo='skip',
            showlegend=False
        ))

    fig.update_layout(
        title=dict(
            text='Prediksi Pergerakan Harga ke Depan & Proyeksi Tren',
            font=dict(size=16, color='#222'),
            y=0.98,
            x=0.5,
            xanchor='center',
            yanchor='top'
        ),
        xaxis=dict(title="Tanggal", showgrid=True, gridcolor='#F0F0F0'),
        yaxis=dict(title=y_label, showgrid=True, gridcolor='#F0F0F0'),
        hovermode='x unified',
        margin=dict(l=40, r=40, t=110, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.04,
            xanchor="center",
            x=0.5
        ),
        template='plotly_white',
        height=480
    )
    return fig

def calculate_vpvr(df, num_bins=28, value_area_pct=0.70):
    """
    Menghitung Volume Profile Visible Range (VPVR) dengan 3 parameter utama:
    1. POC (Point of Control) - Level harga dengan volume transaksi tertinggi
    2. VAH (Value Area High) & VAL (Value Area Low) - Rentang harga 70% total volume
    3. Profil Volume Beli (Up Volume) & Volume Jual (Down Volume) per price bin
    """
    if df is None or df.empty:
        return None
    try:
        close = extract_1d_array(df['Close'] if 'Close' in df.columns else df.iloc[:, 0])
        if len(close) < 2:
            return None
            
        high = extract_1d_array(df['High']) if 'High' in df.columns else close
        low = extract_1d_array(df['Low']) if 'Low' in df.columns else close
        open_p = extract_1d_array(df['Open']) if 'Open' in df.columns else close
        vol = extract_1d_array(df['Volume']) if 'Volume' in df.columns else np.ones_like(close)
        
        vol = np.nan_to_num(vol, nan=1.0)
        if np.all(vol == 0):
            vol = np.ones_like(close)
            
        min_p = float(np.min(low))
        max_p = float(np.max(high))
        if min_p >= max_p:
            max_p = min_p + (min_p * 0.01 if min_p > 0 else 1.0)
            
        bins = np.linspace(min_p, max_p, num_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2.0
        bin_height = float(bins[1] - bins[0])
        
        up_vol = np.zeros(num_bins)
        down_vol = np.zeros(num_bins)
        total_bin_vol = np.zeros(num_bins)
        
        for i in range(len(close)):
            p_c = close[i]
            p_o = open_p[i] if i < len(open_p) else p_c
            v = vol[i] if i < len(vol) else 1.0
            idx = int(np.digitize(p_c, bins) - 1)
            idx = max(0, min(num_bins - 1, idx))
            
            if p_c >= p_o:
                up_vol[idx] += v
            else:
                down_vol[idx] += v
            total_bin_vol[idx] += v
            
        poc_idx = int(np.argmax(total_bin_vol))
        poc_price = float(bin_centers[poc_idx])
        
        # Hitung 70% Value Area (VAH & VAL)
        total_volume = np.sum(total_bin_vol)
        target_va_vol = total_volume * value_area_pct
        
        curr_vol = total_bin_vol[poc_idx]
        up_idx = poc_idx
        down_idx = poc_idx
        
        while curr_vol < target_va_vol and (up_idx < num_bins - 1 or down_idx > 0):
            next_up = total_bin_vol[up_idx + 1] if up_idx < num_bins - 1 else 0
            next_down = total_bin_vol[down_idx - 1] if down_idx > 0 else 0
            
            if next_up >= next_down and up_idx < num_bins - 1:
                up_idx += 1
                curr_vol += next_up
            elif down_idx > 0:
                down_idx -= 1
                curr_vol += next_down
            elif up_idx < num_bins - 1:
                up_idx += 1
                curr_vol += next_up
            else:
                break
                
        val_price = float(bin_centers[down_idx])
        vah_price = float(bin_centers[up_idx])
        max_b_vol = float(np.max(total_bin_vol)) if len(total_bin_vol) > 0 else 1.0
        
        return {
            'bins': bins,
            'bin_centers': bin_centers,
            'bin_height': bin_height,
            'up_vol': up_vol,
            'down_vol': down_vol,
            'total_vol': total_bin_vol,
            'poc_price': poc_price,
            'poc_idx': poc_idx,
            'vah_price': vah_price,
            'val_price': val_price,
            'max_bin_vol': max_b_vol if max_b_vol > 0 else 1.0
        }
    except Exception:
        return None

def plot_comprehensive_market_indicators(df, title, curr_prefix="", asset_type="Crypto"):
    if df is None or df.empty:
        return go.Figure()
        
    dates = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df['Date'] if 'Date' in df.columns else df.index)
    if hasattr(dates, 'tz') and dates.tz is not None:
        dates = dates.tz_localize(None)
        
    close_arr = extract_1d_array(df['Close'] if 'Close' in df.columns else df.iloc[:, 0])
    open_arr = extract_1d_array(df['Open']) if 'Open' in df.columns else close_arr
    high_arr = extract_1d_array(df['High']) if 'High' in df.columns else close_arr
    low_arr = extract_1d_array(df['Low']) if 'Low' in df.columns else close_arr
    
    n = min(len(dates), len(close_arr), len(open_arr), len(high_arr), len(low_arr))
    if n == 0:
        return go.Figure()
        
    dates = dates[:n]
    close_arr = close_arr[:n]
    open_arr = open_arr[:n]
    high_arr = high_arr[:n]
    low_arr = low_arr[:n]
    
    mean_arr = (open_arr + high_arr + low_arr + close_arr) / 4.0
    grand_mean = float(np.mean(close_arr))
    
    vol_arr = extract_1d_array(df['Volume'])[:n] if 'Volume' in df.columns else np.zeros(n)
    atr_arr = calculate_atr_series(df)[:n]
    delta_vol, delta_cols = calculate_daily_delta_volume_series(df)
    delta_vol = delta_vol[:n]
    delta_cols = delta_cols[:n] if len(delta_cols) >= n else None
    vol_bar_cols = get_bar_colors_for_volume(df)
    vol_bar_cols = vol_bar_cols[:n] if vol_bar_cols is not None and len(vol_bar_cols) >= n else None
    
    # Subplot 3 Baris: 1. Candlestick & VPVR, 2. Volume & ATR, 3. Delta Volume Harian
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        row_heights=[0.46, 0.27, 0.27],
        specs=[
            [{"secondary_y": False}],
            [{"secondary_y": True}],
            [{"secondary_y": False}]
        ],
        subplot_titles=[
            f"1. Candlestick {asset_type}, Rata-rata (Mean) & VPVR (Volume Profile: POC, VAH, VAL)",
            "2. Volume Transaksi (Bar) & ATR Volatilitas (Garis Kuning)",
            "3. Delta Volume Harian (Net Buy vs Net Sell)"
        ]
    )
    
    # ----------------------------------------------------
    # Row 1: Candlestick, Mean / Average, and VPVR Profile
    # ----------------------------------------------------
    hover_candle = []
    for i in range(n):
        d_str = format_timestamp_for_plot(dates[i])
        o_s = smart_format(open_arr[i], prefix=curr_prefix)
        h_s = smart_format(high_arr[i], prefix=curr_prefix)
        l_s = smart_format(low_arr[i], prefix=curr_prefix)
        c_s = smart_format(close_arr[i], prefix=curr_prefix)
        m_s = smart_format(mean_arr[i], prefix=curr_prefix)
        v_s = f"{vol_arr[i]:,.0f}" if i < len(vol_arr) else "-"
        chg_candle = ((close_arr[i] - open_arr[i]) / open_arr[i]) * 100.0 if open_arr[i] > 0 else 0.0
        chg_sign = "+" if chg_candle >= 0 else ""
        chg_col = "#00C853" if chg_candle >= 0 else "#D50000"
        hover_candle.append(
            f"<b>Waktu:</b> {d_str}<br>"
            f"<b>Open:</b> {o_s} | <b>Close:</b> {c_s} (<span style='color:{chg_col};'>{chg_sign}{chg_candle:.2f}%</span>)<br>"
            f"<b>High:</b> {h_s} | <b>Low:</b> {l_s}<br>"
            f"<b>Rata-rata (Mean):</b> {m_s}<br>"
            f"<b>Volume:</b> {v_s}"
        )
    
    # 1. Trace Candlestick (Fill Hijau / Merah)
    fig.add_trace(go.Candlestick(
        x=dates,
        open=open_arr,
        high=high_arr,
        low=low_arr,
        close=close_arr,
        name=f'Candlestick {asset_type}',
        increasing_line_color='#00C853',
        increasing_fillcolor='#00C853',
        decreasing_line_color='#D50000',
        decreasing_fillcolor='#D50000',
        hoverinfo='text',
        hovertext=hover_candle
    ), row=1, col=1)
    
    # 2. Garis Mean / Average per Candle (Oranye)
    fig.add_trace(go.Scatter(
        x=dates,
        y=mean_arr,
        mode='lines',
        name='Harga Rata-rata (Mean Candle)',
        line=dict(color='#FF9800', width=1.6, dash='dot'),
        hoverinfo='text',
        hovertext=[f"<b>Waktu:</b> {format_timestamp_for_plot(d)}<br><b>Rata-rata (Mean):</b> {smart_format(m, prefix=curr_prefix)}" for d, m in zip(dates, mean_arr)]
    ), row=1, col=1)
    
    # 3. Garis Horizontal Grand Mean
    fig.add_hline(
        y=grand_mean,
        line_dash="dash",
        line_color="#FFA000",
        line_width=1.3,
        annotation_text=f"Grand Mean: {smart_format(grand_mean, prefix=curr_prefix)}",
        annotation_position="top right",
        annotation_font_size=10,
        annotation_font_color="#FFA000",
        row=1, col=1
    )
    
    # 4. Marker High and Low
    max_idx = int(np.argmax(high_arr)) if len(high_arr) > 0 else -1
    min_idx = int(np.argmin(low_arr)) if len(low_arr) > 0 else -1
    if max_idx >= 0 and min_idx >= 0 and max_idx != min_idx:
        fig.add_trace(go.Scatter(
            x=[dates[max_idx]], y=[high_arr[max_idx]], mode='markers+text', name='Harga Tertinggi (High)',
            marker=dict(color='#00C853', size=9, symbol='triangle-up'),
            text=[f"▲ High: {smart_format(high_arr[max_idx], prefix=curr_prefix)}"],
            textposition="top center", textfont=dict(color='#00C853', size=11),
            hoverinfo='skip', showlegend=False
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=[dates[min_idx]], y=[low_arr[min_idx]], mode='markers+text', name='Harga Terendah (Low)',
            marker=dict(color='#D50000', size=9, symbol='triangle-down'),
            text=[f"▼ Low: {smart_format(low_arr[min_idx], prefix=curr_prefix)}"],
            textposition="bottom center", textfont=dict(color='#D50000', size=11),
            hoverinfo='skip', showlegend=False
        ), row=1, col=1)
        
    # 5. VPVR (Volume Profile Visible Range) di sebelah kiri
    vpvr = calculate_vpvr(df.iloc[:n])
    if vpvr is not None and len(dates) >= 2:
        t_start = dates[0]
        t_end = dates[-1]
        total_span = t_end - t_start
        max_vpvr_width = total_span * 0.18  # 18% dari lebar chart di sisi kiri
        
        vpvr_hover_x = []
        vpvr_hover_y = []
        vpvr_hover_text = []
        
        for k in range(len(vpvr['bin_centers'])):
            b_center = vpvr['bin_centers'][k]
            b_h = vpvr['bin_height'] * 0.42
            b_y0 = b_center - b_h
            b_y1 = b_center + b_h
            
            up_vol_val = vpvr['up_vol'][k]
            down_vol_val = vpvr['down_vol'][k]
            tot_vol_val = vpvr['total_vol'][k]
            
            if tot_vol_val <= 0:
                continue
                
            w_up = (up_vol_val / vpvr['max_bin_vol']) * max_vpvr_width
            w_down = (down_vol_val / vpvr['max_bin_vol']) * max_vpvr_width
            
            x_up_end = t_start + w_up
            x_tot_end = x_up_end + w_down
            
            # Bar Beli (Hijau)
            if w_up > pd.Timedelta(0):
                fig.add_shape(
                    type="rect",
                    x0=t_start, x1=x_up_end,
                    y0=b_y0, y1=b_y1,
                    fillcolor="rgba(0, 200, 83, 0.42)",
                    line=dict(width=0),
                    row=1, col=1
                )
            # Bar Jual (Merah)
            if w_down > pd.Timedelta(0):
                fig.add_shape(
                    type="rect",
                    x0=x_up_end, x1=x_tot_end,
                    y0=b_y0, y1=b_y1,
                    fillcolor="rgba(213, 0, 0, 0.42)",
                    line=dict(width=0),
                    row=1, col=1
                )
                
            vpvr_hover_x.append(t_start + (w_up + w_down) / 2.0)
            vpvr_hover_y.append(b_center)
            vpvr_hover_text.append(
                f"<b>[VPVR Volume Profile]</b><br>"
                f"<b>Tingkat Harga:</b> {smart_format(b_center, prefix=curr_prefix)}<br>"
                f"<b>Volume Beli (Up):</b> {up_vol_val:,.0f}<br>"
                f"<b>Volume Jual (Down):</b> {down_vol_val:,.0f}<br>"
                f"<b>Total Volume:</b> {tot_vol_val:,.0f}"
            )
            
        if vpvr_hover_x:
            fig.add_trace(go.Scatter(
                x=vpvr_hover_x,
                y=vpvr_hover_y,
                mode='markers',
                marker=dict(size=1, opacity=0),
                name='VPVR Volume Profile (Beli 🟢 / Jual 🔴)',
                hoverinfo='text',
                hovertext=vpvr_hover_text,
                showlegend=True
            ), row=1, col=1)

        # Garis POC (Point of Control)
        fig.add_shape(
            type="line",
            x0=t_start, x1=t_end,
            y0=vpvr['poc_price'], y1=vpvr['poc_price'],
            line=dict(color="#FF1744", width=2.0, dash="solid"),
            row=1, col=1
        )
        fig.add_annotation(
            x=t_end, y=vpvr['poc_price'],
            text=f"POC: {smart_format(vpvr['poc_price'], prefix=curr_prefix)}",
            showarrow=False,
            font=dict(color="#FF1744", size=10),
            bgcolor="rgba(255, 255, 255, 0.8)",
            xanchor="right", yanchor="bottom",
            row=1, col=1
        )

        # Garis VAH & VAL (Value Area High & Low 70%)
        fig.add_shape(
            type="line",
            x0=t_start, x1=t_end,
            y0=vpvr['vah_price'], y1=vpvr['vah_price'],
            line=dict(color="#0288D1", width=1.5, dash="dash"),
            row=1, col=1
        )
        fig.add_annotation(
            x=t_end, y=vpvr['vah_price'],
            text=f"VAH (70%): {smart_format(vpvr['vah_price'], prefix=curr_prefix)}",
            showarrow=False,
            font=dict(color="#0288D1", size=9),
            bgcolor="rgba(255, 255, 255, 0.7)",
            xanchor="right", yanchor="bottom",
            row=1, col=1
        )

        fig.add_shape(
            type="line",
            x0=t_start, x1=t_end,
            y0=vpvr['val_price'], y1=vpvr['val_price'],
            line=dict(color="#0288D1", width=1.5, dash="dash"),
            row=1, col=1
        )
        fig.add_annotation(
            x=t_end, y=vpvr['val_price'],
            text=f"VAL (70%): {smart_format(vpvr['val_price'], prefix=curr_prefix)}",
            showarrow=False,
            font=dict(color="#0288D1", size=9),
            bgcolor="rgba(255, 255, 255, 0.7)",
            xanchor="right", yanchor="top",
            row=1, col=1
        )
        
    # ----------------------------------------------------
    # Row 2: Volume (Bar) and ATR (Line on secondary y-axis)
    # ----------------------------------------------------
    vol_max_idx = int(np.argmax(vol_arr)) if len(vol_arr) > 0 and not np.all(vol_arr == 0) else -1
    vol_min_idx = int(np.argmin(vol_arr)) if len(vol_arr) > 0 and not np.all(vol_arr == 0) else -1
    
    if len(vol_arr) > 0 and not np.all(vol_arr == 0):
        hover_vol = []
        for i in range(n):
            d_str = format_timestamp_for_plot(dates[i])
            v_val = vol_arr[i]
            tag = ""
            if i == vol_max_idx and vol_max_idx != vol_min_idx:
                tag = '<br><b style="color:#00C853;">▲ [Volume Tertinggi (High)]</b>'
            elif i == vol_min_idx and vol_max_idx != vol_min_idx:
                tag = '<br><b style="color:#D50000;">▼ [Volume Terendah (Low)]</b>'
            hover_vol.append(f"<b>Waktu:</b> {d_str}<br><b>Volume:</b> {v_val:,.0f}{tag}")

        if n > 200:
            fig.add_trace(go.Scatter(
                x=dates, y=vol_arr, mode='lines',
                line=dict(color='rgba(0, 200, 83, 0.4)', width=1),
                fill='tozeroy', fillcolor='rgba(0, 200, 83, 0.15)',
                name='Volume Area', showlegend=False, hoverinfo='skip'
            ), row=2, col=1, secondary_y=False)

        fig.add_trace(go.Bar(
            x=dates, y=vol_arr, name='Volume',
            marker_color=vol_bar_cols if vol_bar_cols else '#00C853',
            opacity=1.0,
            hoverinfo='text',
            hovertext=hover_vol
        ), row=2, col=1, secondary_y=False)

        # High & Low Markers for Volume
        if vol_max_idx >= 0 and vol_min_idx >= 0 and vol_max_idx != vol_min_idx:
            fig.add_trace(go.Scatter(
                x=[dates[vol_max_idx]], y=[vol_arr[vol_max_idx]], mode='markers+text', name='Volume Tertinggi',
                marker=dict(color='#00C853', size=8, symbol='triangle-up'),
                text=[f"▲ Vol High: {vol_arr[vol_max_idx]:,.0f}"],
                textposition="top left", textfont=dict(color='#00C853', size=10),
                hoverinfo='skip', showlegend=False
            ), row=2, col=1, secondary_y=False)
            fig.add_trace(go.Scatter(
                x=[dates[vol_min_idx]], y=[vol_arr[vol_min_idx]], mode='markers+text', name='Volume Terendah',
                marker=dict(color='#D50000', size=8, symbol='triangle-down'),
                text=[f"▼ Vol Low: {vol_arr[vol_min_idx]:,.0f}"],
                textposition="top right", textfont=dict(color='#D50000', size=10),
                hoverinfo='skip', showlegend=False
            ), row=2, col=1, secondary_y=False)
        
    atr_max_idx = int(np.argmax(atr_arr)) if len(atr_arr) == n and len(atr_arr) > 0 else -1
    atr_min_idx = int(np.argmin(atr_arr)) if len(atr_arr) == n and len(atr_arr) > 0 else -1

    if len(atr_arr) == n:
        hover_atr = []
        for i in range(n):
            d_str = format_timestamp_for_plot(dates[i])
            a_val = atr_arr[i]
            tag = ""
            if i == atr_max_idx and atr_max_idx != atr_min_idx:
                tag = '<br><b style="color:#FFB300;">▲ [ATR Tertinggi (High)]</b>'
            elif i == atr_min_idx and atr_max_idx != atr_min_idx:
                tag = '<br><b style="color:#FF6D00;">▼ [ATR Terendah (Low)]</b>'
            hover_atr.append(f"<b>Waktu:</b> {d_str}<br><b>ATR:</b> {smart_format(a_val, prefix=curr_prefix)}{tag}")

        fig.add_trace(go.Scatter(
            x=dates, y=atr_arr, mode='lines', name='ATR (Volatilitas)',
            line=dict(color='#FFB300', width=2.0),
            hoverinfo='text',
            hovertext=hover_atr
        ), row=2, col=1, secondary_y=True)

        # High & Low Markers for ATR
        if atr_max_idx >= 0 and atr_min_idx >= 0 and atr_max_idx != atr_min_idx:
            fig.add_trace(go.Scatter(
                x=[dates[atr_max_idx]], y=[atr_arr[atr_max_idx]], mode='markers+text', name='ATR Tertinggi',
                marker=dict(color='#FFB300', size=8, symbol='triangle-up'),
                text=[f"▲ ATR High: {smart_format(atr_arr[atr_max_idx], prefix=curr_prefix)}"],
                textposition="top right", textfont=dict(color='#FFB300', size=10),
                hoverinfo='skip', showlegend=False
            ), row=2, col=1, secondary_y=True)
            fig.add_trace(go.Scatter(
                x=[dates[atr_min_idx]], y=[atr_arr[atr_min_idx]], mode='markers+text', name='ATR Terendah',
                marker=dict(color='#FF6D00', size=8, symbol='triangle-down'),
                text=[f"▼ ATR Low: {smart_format(atr_arr[atr_min_idx], prefix=curr_prefix)}"],
                textposition="top left", textfont=dict(color='#FF6D00', size=10),
                hoverinfo='skip', showlegend=False
            ), row=2, col=1, secondary_y=True)
        
    # ----------------------------------------------------
    # Row 3: Delta Volume (Bar)
    # ----------------------------------------------------
    if len(delta_vol) > 0:
        fig.add_hline(y=0, line_dash="solid", line_color="#E0E0E0", line_width=1, row=3, col=1)

        delta_max_idx = int(np.argmax(delta_vol))
        delta_min_idx = int(np.argmin(delta_vol))

        hover_delta = []
        for i in range(len(delta_vol)):
            d_str = format_timestamp_for_plot(dates[i])
            dv = delta_vol[i]
            sign = "+" if dv >= 0 else ""
            tag = ""
            if i == delta_max_idx and delta_max_idx != delta_min_idx:
                tag = '<br><b style="color:#00C853;">▲ [Delta Tertinggi (Net Buy Max)]</b>'
            elif i == delta_min_idx and delta_max_idx != delta_min_idx:
                tag = '<br><b style="color:#D50000;">▼ [Delta Terendah (Net Sell Max)]</b>'
            hover_delta.append(f"<b>Waktu:</b> {d_str}<br><b>Delta:</b> {sign}{dv:,.0f}{tag}")

        if n > 200:
            fig.add_trace(go.Scatter(
                x=dates, y=delta_vol, mode='lines',
                line=dict(color='rgba(120, 120, 120, 0.35)', width=1),
                name='Delta Curve', showlegend=False, hoverinfo='skip'
            ), row=3, col=1)

        fig.add_trace(go.Bar(
            x=dates, y=delta_vol, name='Delta Volume',
            marker_color=delta_cols if delta_cols else '#00C853',
            opacity=1.0,
            hoverinfo='text',
            hovertext=hover_delta
        ), row=3, col=1)

        # High & Low Markers for Delta Volume
        if delta_max_idx != delta_min_idx:
            fig.add_trace(go.Scatter(
                x=[dates[delta_max_idx]], y=[delta_vol[delta_max_idx]], mode='markers+text', name='Delta Tertinggi',
                marker=dict(color='#00C853', size=8, symbol='triangle-up'),
                text=[f"▲ Delta High: {('+' if delta_vol[delta_max_idx] >= 0 else '')}{delta_vol[delta_max_idx]:,.0f}"],
                textposition="top center", textfont=dict(color='#00C853', size=10),
                hoverinfo='skip', showlegend=False
            ), row=3, col=1)
            fig.add_trace(go.Scatter(
                x=[dates[delta_min_idx]], y=[delta_vol[delta_min_idx]], mode='markers+text', name='Delta Terendah',
                marker=dict(color='#D50000', size=8, symbol='triangle-down'),
                text=[f"▼ Delta Low: {delta_vol[delta_min_idx]:,.0f}"],
                textposition="bottom center", textfont=dict(color='#D50000', size=10),
                hoverinfo='skip', showlegend=False
            ), row=3, col=1)
        
    # ----------------------------------------------------
    # Layout & Y-Axes Padding Settings (Fixes all text clipping)
    # ----------------------------------------------------
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color='#222'), y=0.985, x=0.5, xanchor='center', yanchor='top'),
        hovermode='x unified',
        margin=dict(l=60, r=60, t=80, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        template='plotly_white',
        height=980,
        showlegend=True,
        xaxis_rangeslider_visible=False
    )
    
    # 1. Padding Row 1 (Candlestick & VPVR)
    p_min = float(np.min(low_arr))
    p_max = float(np.max(high_arr))
    p_span = (p_max - p_min) if p_max > p_min else (p_max * 0.05 if p_max > 0 else 1.0)
    fig.update_yaxes(title_text=f"Harga {asset_type}", range=[max(0.0, p_min - p_span * 0.18), p_max + p_span * 0.18], row=1, col=1)
    
    # 2. Padding Row 2 (Volume & ATR)
    if len(vol_arr) > 0 and not np.all(vol_arr == 0):
        v_max = float(np.max(vol_arr))
        fig.update_yaxes(title_text="Volume", range=[0, v_max * 1.55 if v_max > 0 else 1.0], row=2, col=1, secondary_y=False)
    else:
        fig.update_yaxes(title_text="Volume", row=2, col=1, secondary_y=False)
        
    if len(atr_arr) == n:
        a_min = float(np.min(atr_arr))
        a_max = float(np.max(atr_arr))
        a_span = (a_max - a_min) if a_max > a_min else (a_max * 0.1 if a_max > 0 else 1.0)
        fig.update_yaxes(title_text="ATR", range=[max(0.0, a_min - a_span * 0.35), a_max + a_span * 0.50], row=2, col=1, secondary_y=True)
    else:
        fig.update_yaxes(title_text="ATR", row=2, col=1, secondary_y=True)
        
    # 3. Padding Row 3 (Delta Volume)
    if len(delta_vol) > 0:
        d_min = float(np.min(delta_vol))
        d_max = float(np.max(delta_vol))
        d_span = max(abs(d_min), abs(d_max))
        if d_span == 0:
            d_span = 1.0
        fig.update_yaxes(title_text="Net Delta", range=[d_min - d_span * 0.50, d_max + d_span * 0.50], row=3, col=1)
    else:
        fig.update_yaxes(title_text="Net Delta", row=3, col=1)
        
    fig.update_xaxes(title_text="Waktu / Tanggal", row=3, col=1)
    return fig

def get_metric_badge_info(category):
    colors = {
        'sangat_baik': ('#00C853', 'rgba(0, 200, 83, 0.08)', '🟢 Sangat Baik'),
        'baik': ('#FFB300', 'rgba(255, 179, 0, 0.08)', '🟡 Baik'),
        'cukup_baik': ('#00B0FF', 'rgba(0, 176, 255, 0.08)', '🔵 Cukup Baik'),
        'kurang_baik': ('#FF6D00', 'rgba(255, 109, 0, 0.08)', '🟠 Kurang Baik'),
        'buruk': ('#D50000', 'rgba(213, 0, 0, 0.08)', '🔴 Buruk'),
        'neutral': ('#9E9E9E', 'rgba(158, 158, 158, 0.08)', '⚪ N/A')
    }
    return colors.get(category, colors['neutral'])

def render_colored_metric_card(label, value_str, category, subtext=""):
    border_color, bg_color, badge_text = get_metric_badge_info(category)
    subtext_html = f'<div style="font-size: 11px; color: #777; margin-top: 2px;">{subtext}</div>' if subtext else ''
    html_code = f"""
    <div style="background-color: {bg_color}; border-left: 4px solid {border_color}; border-radius: 8px; padding: 10px 12px; margin-bottom: 10px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <span style="font-size: 12px; color: #555; font-weight: 600;">{label}</span>
            <span style="font-size: 11px; font-weight: 700; color: {border_color};">{badge_text}</span>
        </div>
        <div style="font-size: 19px; font-weight: 700; color: #222; margin-top: 4px;">{value_str}</div>
        {subtext_html}
    </div>
    """
    st.markdown(html_code, unsafe_allow_html=True)

def evaluate_model_performance(y_test, y_pred):
    n_samples = len(y_test)
    if n_samples == 0:
        return None

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    accuracy = 100 - (mape * 100)
    
    mean_actual = np.mean(y_test)
    nrmse = (rmse / mean_actual) if mean_actual > 0 else 1.0
    nrmse_pct = nrmse * 100.0
    
    # R2 Score calculation (hanya valid jika sampel >= 8 dan varians data mencukupi)
    if n_samples >= 8:
        try:
            var_y = np.var(y_test)
            if var_y > 1e-6:
                r2 = r2_score(y_test, y_pred)
                if np.isnan(r2) or np.isinf(r2):
                    r2 = None
            else:
                r2 = None
        except Exception:
            r2 = None
    else:
        r2 = None

    # Akurasi Arah Pergerakan (Mean Directional Accuracy) untuk sampel >= 2
    if n_samples >= 2:
        actual_dir = np.diff(y_test) >= 0
        pred_dir = np.diff(y_pred) >= 0
        mda = np.mean(actual_dir == pred_dir) * 100.0
    else:
        mda = None

    # 1. Klasifikasi Warna & Skor Akurasi
    if accuracy >= 95:
        cat_acc = 'sangat_baik'
        score_acc = 100
    elif accuracy >= 85:
        cat_acc = 'baik'
        score_acc = 80 + (accuracy - 85) * 2
    elif accuracy >= 75:
        cat_acc = 'cukup_baik'
        score_acc = 60 + (accuracy - 75) * 2
    elif accuracy >= 50:
        cat_acc = 'kurang_baik'
        score_acc = 40 + (accuracy - 50) * (20 / 25)
    elif accuracy > 0:
        cat_acc = 'buruk'
        score_acc = (accuracy / 50) * 40
    else:
        cat_acc = 'buruk'
        score_acc = 0

    # 2. Klasifikasi Warna MAPE
    if mape <= 0.05:
        cat_mape = 'sangat_baik'
    elif mape <= 0.15:
        cat_mape = 'baik'
    elif mape <= 0.25:
        cat_mape = 'cukup_baik'
    elif mape <= 0.50:
        cat_mape = 'kurang_baik'
    else:
        cat_mape = 'buruk'

    # 3. Klasifikasi Warna NRMSE (juga menjadi basis warna RMSE, MAE, MSE)
    if nrmse <= 0.05:
        cat_nrmse = 'sangat_baik'
        score_nrmse = 100
    elif nrmse <= 0.10:
        cat_nrmse = 'baik'
        score_nrmse = 80 + (0.10 - nrmse) * (20 / 0.05)
    elif nrmse <= 0.20:
        cat_nrmse = 'cukup_baik'
        score_nrmse = 60 + (0.20 - nrmse) * (20 / 0.10)
    elif nrmse <= 0.35:
        cat_nrmse = 'kurang_baik'
        score_nrmse = 40 + (0.35 - nrmse) * (20 / 0.15)
    else:
        cat_nrmse = 'buruk'
        score_nrmse = max(0, 40 - (nrmse - 0.35) * 40)

    cat_rmse = cat_nrmse
    cat_mse = cat_nrmse
    cat_mae = cat_nrmse

    # 4. Klasifikasi Warna & Skor R2
    if r2 is not None:
        if r2 >= 0.90:
            cat_r2 = 'sangat_baik'
            score_r2 = 100
        elif r2 >= 0.75:
            cat_r2 = 'baik'
            score_r2 = 80 + (r2 - 0.75) * (20 / 0.15)
        elif r2 >= 0.50:
            cat_r2 = 'cukup_baik'
            score_r2 = 60 + (r2 - 0.50) * (20 / 0.25)
        elif r2 >= 0.0:
            cat_r2 = 'kurang_baik'
            score_r2 = 40 + (r2) * (20 / 0.50)
        else:
            cat_r2 = 'buruk'
            score_r2 = max(0, 40 + r2 * 20)
    else:
        cat_r2 = 'neutral'
        score_r2 = score_acc

    # 5. Klasifikasi MDA
    if mda is not None:
        if mda >= 70:
            cat_mda = 'sangat_baik'
        elif mda >= 55:
            cat_mda = 'baik'
        elif mda >= 45:
            cat_mda = 'cukup_baik'
        else:
            cat_mda = 'kurang_baik'
    else:
        cat_mda = 'neutral'

    # Akumulasi Skor Gabungan:
    if n_samples >= 10 and r2 is not None:
        composite_score = 0.40 * score_acc + 0.35 * score_r2 + 0.25 * score_nrmse
    else:
        # Horizon singkat (1-9 hari): mengutamakan Akurasi & Rasio Error Nominal
        composite_score = 0.60 * score_acc + 0.40 * score_nrmse

    if accuracy < 0:
        label = "Performa: Sangat Buruk (Akurasi Negatif)"
        status = "error" # Merah
        badge = "🔴"
    elif composite_score >= 88:
        label = "Performa: Sangat Baik"
        status = "success" # Hijau
        badge = "🟢"
    elif composite_score >= 75:
        label = "Performa: Baik"
        status = "success" # Hijau/Kuning
        badge = "🟡"
    elif composite_score >= 60:
        label = "Performa: Cukup Baik"
        status = "info" # Biru
        badge = "🔵"
    elif composite_score >= 40:
        label = "Performa: Kurang Baik"
        status = "warning" # Oranye
        badge = "🟠"
    else:
        label = "Performa: Sangat Buruk"
        status = "error" # Merah
        badge = "🔴"

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "mape": mape,
        "accuracy": accuracy,
        "nrmse": nrmse,
        "nrmse_pct": nrmse_pct,
        "mda": mda,
        "composite_score": composite_score,
        "cat_acc": cat_acc,
        "cat_mape": cat_mape,
        "cat_r2": cat_r2,
        "cat_rmse": cat_rmse,
        "cat_mse": cat_mse,
        "cat_mae": cat_mae,
        "cat_nrmse": cat_nrmse,
        "cat_mda": cat_mda,
        "label": label,
        "status": status,
        "badge": badge
    }

def main(stock, data_source="yfinance", api_key=""):
    if 'current_stock' not in st.session_state:
        st.session_state.current_stock = ""
    if 'training_completed' not in st.session_state:
        st.session_state.training_completed = False

    state_key = f"{stock}_{data_source}"
    if st.session_state.current_stock != state_key:
        if st.session_state.training_completed:
            st.cache_data.clear()
            st.cache_resource.clear()
            st.session_state.training_completed = False
        st.session_state.current_stock = state_key

    is_cmc = (data_source == "coinmarketcap")
    clean_sym = stock.replace('-USD', '').replace('-IDR', '').replace(' ', '').upper()
    asset_type = "Crypto" if (is_cmc or is_crypto_ticker(stock)) else "Saham"

    if is_cmc:
        st.header(f"Prediksi Harga {clean_sym} melalui Coinmarketcap (CNN-GRU)")
    else:
        st.header(f"Prediksi Harga {asset_type} dengan kode {stock}")

    # Ringkasan 3 kolom di bawah header
    curr_prefix = "$ " if (is_cmc or stock.endswith('-USD') or stock.endswith('-IDR') or 'USD' in stock) else "Rp "
    last_vol = 0.0
    last_pr = 0.0
    try:
        if is_cmc:
            quick_df = fetch_coinmarketcap_data(clean_sym, api_key=api_key, interval="1m", count=1000)
        elif crypto_yfinance and is_crypto_ticker(stock):
            quick_df = cyf.download(stock, start="2020-01-01", end=date.today().strftime("%Y-%m-%d"))
        else:
            quick_df = yf.download(stock, start="2020-01-01", end=date.today().strftime("%Y-%m-%d"))
        
        if not quick_df.empty:
            quick_df = ensure_datetime_index(quick_df)
            last_dt = format_timestamp_for_plot(quick_df.index[-1])
            last_pr = safe_float(quick_df['Close'].iloc[-1])
            last_pr_str = smart_format(last_pr, prefix=curr_prefix)
            if 'Volume' in quick_df.columns:
                last_vol = safe_float(quick_df['Volume'].iloc[-1])
        else:
            last_dt = "-"
            last_pr_str = "-"
    except Exception:
        last_dt = "-"
        last_pr_str = "-"

    mcap_val = get_market_cap(stock, last_close=last_pr, last_volume=last_vol)
    mcap_str = format_market_cap(mcap_val, curr_prefix=curr_prefix)

    col_sum1, col_sum2, col_sum3 = st.columns(3)
    with col_sum1:
        st.metric("Tanggal / Waktu Terakhir", last_dt)
    with col_sum2:
        st.metric("Harga Terakhir", last_pr_str)
    with col_sum3:
        st.metric("Market Cap", mcap_str)

    # Mini Trend Metrics: If CMC show 10 metrics (1m, 5m, 30m, 1H, 12H, 1D, 1W, 1M, 90D, YTD)
    if not quick_df.empty and len(quick_df) >= 2:
        if is_cmc:
            render_coinmarketcap_mini_metrics(quick_df, clean_sym)
        else:
            c_now = safe_float(quick_df['Close'].iloc[-1])
            c_prev = safe_float(quick_df['Close'].iloc[-2])
            chg_1d = ((c_now - c_prev) / c_prev) * 100 if c_prev > 0 else 0.0
            chg_1w = get_change_pct(quick_df, 7)
            chg_1m = get_change_pct(quick_df, 30)
            chg_90d = get_change_pct(quick_df, 90)
            chg_ytd = get_change_pct(quick_df, 365)

            st.markdown(f"**Performa Perubahan Harga {stock}:**")
            c1, c2, c3, c4, c5 = st.columns(5)
            metrics_list = [
                (c1, "1 Hari (1D)", chg_1d),
                (c2, "1 Minggu (1W)", chg_1w),
                (c3, "1 Bulan (1M)", chg_1m),
                (c4, "90 Hari (90D)", chg_90d),
                (c5, "1 Tahun (YTD)", chg_ytd)
            ]
            for col, label, val in metrics_list:
                with col:
                    if val is not None and not pd.isna(val):
                        color = "#00C853" if val >= 0 else "#D50000"
                        symbol = "▲" if val >= 0 else "▼"
                        sign = "+" if val > 0 else ""
                        bg_color = "rgba(0, 200, 83, 0.08)" if val >= 0 else "rgba(213, 0, 0, 0.08)"
                        st.markdown(f"""
                        <div style="background-color: {bg_color}; padding: 8px 6px; border-radius: 8px; border-left: 4px solid {color}; text-align: center;">
                            <div style="font-size: 11px; color: #555; font-weight: 600;">{label}</div>
                            <div style="font-size: 15px; font-weight: bold; color: {color}; margin-top: 2px;">{symbol} {sign}{val:.2f}%</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="background-color: #f5f5f5; padding: 8px 6px; border-radius: 8px; text-align: center;">
                            <div style="font-size: 11px; color: #777; font-weight: 600;">{label}</div>
                            <div style="font-size: 15px; font-weight: bold; color: #999; margin-top: 2px;">-</div>
                        </div>
                        """, unsafe_allow_html=True)
    st.write("")

    with st.expander("1. Persiapan Lingkungan"):
        with st.spinner("Mengimpor library yang diperlukan..."):
            with st.popover("Detail Library yang Digunakan"):
                st.info('Import Library Utama: `time`, `numpy`, `pandas`, `yfinance`, `PIL` untuk pengelolaan data dasar, manipulasi array, serta penampilan gambar.', icon=":material/code:")
                st.warning('Import Framework ML & Visualisasi: `streamlit`, `tensorflow`, `matplotlib`, `sklearn`, `streamlit_option_menu` untuk pemodelan CNN-GRU dan antarmuka web interaktif.', icon=":material/extension:")
            
            with st.popover("📊 Diagram Alur Kerja (Workflow)"):
                mermaid_code = """
                graph TD
                    A([🚀 Input Kode Saham / Kripto]) --> B{Deteksi Jenis Aset / Sumber Data}
                    B -->|yFinance Saham / Crypto| C1[Scraping Data via yfinance]
                    B -->|CoinMarketCap API| C2[Scraping Intraday via CoinMarketCap Pro API]
                    C1 --> D[Ringkasan Metrik Pasar, Performa & Grafik Mini Sparkline]
                    C2 --> D
                    D --> E[2. Pengumpulan Data: Rentang Waktu Historis Pelatihan]
                    E --> F[3. Pra-pemrosesan Data: Normalisasi MinMaxScaler & Windowing Lookback]
                    F --> G[4. Perancangan Model: Conv1D + GRU + Dropout + Dense]
                    G --> H[5. Pelatihan Model: Adam Optimizer & Loss Function MSE]
                    H --> I[6. Evaluasi Model Multi-Metrik: Akurasi, MAPE, RMSE, MAE, R2, MDA]
                    I --> J{Evaluasi Kualitas Model}
                    J -->|Akurasi Positif & Optimal| K[7. Visualisasi Prediksi Multi-Horizon]
                    J -->|Akurasi Rendah / Negatif| L[Saran Penyesuaian Epoch / Batch / Data]
                    K --> M[8. Interpretasi Tren Pasar & Rekomendasi Investasi]
                    L --> E
                    M --> N([🎯 Selesai / Ganti Kode Aset & Auto Reset Cache])
                """
                st.code(mermaid_code, language='mermaid')
            
            st.success("Library berhasil diimpor")

    with st.expander("2. Pengumpulan Data"):

        @st.cache_data
        def load_data(ticker, start_date, end_date):
            try:
                if crypto_yfinance and is_crypto_ticker(ticker):
                    data = cyf.download(ticker, start=start_date, end=end_date)
                else:
                    data = yf.download(ticker, start=start_date, end=end_date)
                
                if data.empty:
                    st.error(f"Tidak dapat memuat data untuk {ticker}. Silakan coba simbol lain.")
                    return pd.DataFrame()
                    
                data = ensure_datetime_index(data)
                return data
            except Exception as e:
                st.error(f"Error loading data for {ticker}: {str(e)}")
                return pd.DataFrame()

        # DATA HISTORY
        if is_cmc:
            full_data = quick_df.copy()
        else:
            full_data = load_data(stock, "2000-01-01", date.today().strftime("%Y-%m-%d")).copy()

        if full_data.empty or len(full_data) == 0:
            st.error(f"❌ Simbol ticker **'{stock}'** tidak ditemukan atau data historis tidak tersedia.", icon=":material/error:")
            st.info("""
            💡 **Panduan Format Input Ticker yang Benar:**
            - **CoinMarketCap**: Masukkan simbol koin langsung (contoh: `BTC`, `ETH`, `DOGE`, `SOL`, `BNB`, `XRP`).
            - **Aset Kripto (yFinance)**: Tambahkan `-USD` di akhir simbol (contoh: `BTC-USD`, `ETH-USD`, `DOGE-USD`).
            - **Saham Indonesia (BEI / IDX)**: Tambahkan `.JK` di akhir kode emiten (contoh: `BBCA.JK`, `BMRI.JK`, `TLKM.JK`).
            - **Saham Global (Wall Street / US)**: Masukkan kode ticker langsung (contoh: `AAPL`, `NVDA`, `TSLA`, `MSFT`).
            """, icon=":material/lightbulb:")
            st.stop()

        st.subheader("Data keseluruhan")
        st.write("Mulai")
        st.write(format_df_for_display(full_data.head(1)))
        st.write("Hingga")
        st.write(format_df_for_display(full_data.tail(1)))

        if not full_data.empty and len(full_data) >= 2:
            c_start_f = safe_float(full_data['Close'].iloc[0])
            c_end_f = safe_float(full_data['Close'].iloc[-1])
            tot_chg_f = ((c_end_f - c_start_f) / c_start_f) * 100.0 if c_start_f > 0 else 0.0
            tot_sign_f = f":green[▲ +{tot_chg_f:,.2f}%]" if tot_chg_f >= 0 else f":red[▼ {tot_chg_f:,.2f}%]"
            tot_diff_f = c_end_f - c_start_f
            diff_sign_f = "+" if tot_diff_f >= 0 else ""
            tot_diff_str_f = smart_format(tot_diff_f, prefix=curr_prefix)
            full_chart_color = '#00C853' if tot_chg_f >= 0 else '#D50000'
            st.markdown(f"**Performa Perubahan Keseluruhan Data:** {tot_sign_f} (`{diff_sign_f}{tot_diff_str_f}`)")
        else:
            full_chart_color = '#00C853'

        # Plot Interaktif dengan Plotly untuk data keseluruhan (Warna dinamis Hijau Naik / Merah Turun)
        fig1 = plot_interactive_history(full_data, f'Data Keseluruhan Harga {asset_type}', f'Harga {asset_type}', full_chart_color, curr_prefix=curr_prefix)
        if fig1 is not None:
            render_plotly_with_tools(fig1, key="fig_full_data_history")

        with st.expander(f"📈 Grafik Tren Riwayat Harga Candlestick, VPVR, Volume, ATR & Delta Volume (Data Keseluruhan)", expanded=False):
            fig_full_comp = plot_comprehensive_market_indicators(full_data, f'Indikator Pasar Komprehensif Candlestick & VPVR (Data Keseluruhan {asset_type})', curr_prefix=curr_prefix, asset_type=asset_type)
            if fig_full_comp is not None:
                render_plotly_with_tools(fig_full_comp, key="fig_full_data_comp_indicators")

        with st.popover("Tampilkan Semua Data"):
            st.write(format_df_for_display(full_data))
            csv_full = full_data.to_csv().encode('utf-8')
            st.download_button(
                label="📥 Unduh Data Keseluruhan sebagai CSV",
                data=csv_full,
                file_name=f"data_keseluruhan_{stock}.csv",
                mime="text/csv",
                key="btn_dl_full_data"
            )

        # DATA PELATIHAN
        st.subheader("Pengaturan Data Pelatihan")

        # Map durasi preset (Tombol & Radiobox Cepat)
        PRESET_DURATIONS = {
            "1 m": timedelta(minutes=1),
            "3 m": timedelta(minutes=3),
            "5 m": timedelta(minutes=5),
            "15 m": timedelta(minutes=15),
            "30 m": timedelta(minutes=30),
            "1 H": timedelta(hours=1),
            "4 H": timedelta(hours=4),
            "1 D": timedelta(days=1),
            "1 W": timedelta(days=7),
            "1 M": timedelta(days=30),
            "90 D": timedelta(days=90),
            "YTD": timedelta(days=365),
            "2 T": timedelta(days=365*2),
            "3 T": timedelta(days=365*3),
            "4 T": timedelta(days=365*4),
            "5 T": timedelta(days=365*5),
            "10 T": timedelta(days=365*10),
            "30 T": timedelta(days=365*30)
        }

        def format_quick_button_label(lbl):
            clean = lbl.strip()
            if clean in ["1 H", "1H"]:
                return ":orange[**1 H**]"
            elif clean in ["4 H", "4H"]:
                return ":red[**4 H**]"
            elif clean in ["1 D", "1D"]:
                return ":green[**1 D**]"
            return lbl

        END_TIME_OFFSETS = [
            ("1 m", timedelta(minutes=1)),
            ("3 m", timedelta(minutes=3)),
            ("5 m", timedelta(minutes=5)),
            ("15 m", timedelta(minutes=15)),
            ("30 m", timedelta(minutes=30)),
            ("1 H", timedelta(hours=1)),
            ("4 H", timedelta(hours=4)),
            ("1 D", timedelta(days=1)),
            ("1 W", timedelta(days=7)),
            ("1 M", timedelta(days=30)),
            ("90 D", timedelta(days=90)),
            ("YTD", timedelta(days=365))
        ]

        if is_cmc:
            if 'cmc_end_offset_btn' not in st.session_state:
                st.session_state.cmc_end_offset_btn = None
            if 'cmc_preset_selected' not in st.session_state:
                st.session_state.cmc_preset_selected = "30 T"

            # 1. Pilihan Tanggal / Waktu Selesai (Hari ini / Uncheck Kalender & Tombol)
            use_today_end = st.checkbox("Gunakan hingga tanggal terbaru (Hari ini)", value=True, key="chk_use_today_cmc")
            latest_avail_dt = full_data.index[-1] if not full_data.empty else pd.Timestamp.now()

            if use_today_end:
                end_datetime = latest_avail_dt
                st.session_state.cmc_end_offset_btn = None
            else:
                st.markdown("**Pilih Tanggal / Waktu Selesai (Data Historis):**")
                st.markdown("<small><b>Tentukan Titik Akhir Berdasarkan Waktu Tersedia dari Data Terakhir (Khusus 1H 🟠, 4H 🔴, 1D 🟢):</b></small>", unsafe_allow_html=True)
                
                # Tombol Cepat Titik Akhir (1m - YTD)
                btn_cols_1 = st.columns(6)
                btn_cols_2 = st.columns(6)
                for b_i, (t_lbl, t_delta) in enumerate(END_TIME_OFFSETS[:6]):
                    with btn_cols_1[b_i]:
                        lbl_styled = format_quick_button_label(t_lbl)
                        b_type = "primary" if st.session_state.cmc_end_offset_btn == t_lbl else "secondary"
                        if st.button(lbl_styled, key=f"btn_end_cmc_{t_lbl}", type=b_type, use_container_width=True):
                            st.session_state.cmc_end_offset_btn = t_lbl
                            st.rerun()
                for b_i, (t_lbl, t_delta) in enumerate(END_TIME_OFFSETS[6:]):
                    with btn_cols_2[b_i]:
                        lbl_styled = format_quick_button_label(t_lbl)
                        b_type = "primary" if st.session_state.cmc_end_offset_btn == t_lbl else "secondary"
                        if st.button(lbl_styled, key=f"btn_end_cmc_{t_lbl}", type=b_type, use_container_width=True):
                            st.session_state.cmc_end_offset_btn = t_lbl
                            st.rerun()

                selected_end_offset = st.session_state.cmc_end_offset_btn
                if selected_end_offset:
                    offset_dict = dict(END_TIME_OFFSETS)
                    delta_val = offset_dict.get(selected_end_offset, timedelta(0))
                    end_datetime = latest_avail_dt - delta_val
                    st.info(f"📍 Titik Akhir Terpilih via Tombol: **{selected_end_offset} yang lalu** (Per: `{format_timestamp_for_plot(end_datetime)}`).")
                else:
                    default_end_d = latest_avail_dt.date() if hasattr(latest_avail_dt, 'date') else date.today()
                    end_date_selected = st.date_input(
                        "📅 Tanggal Selesai Pelatihan (Kalender):",
                        value=default_end_d,
                        min_value=date(2010, 1, 1),
                        max_value=latest_avail_dt.date() if hasattr(latest_avail_dt, 'date') else date.today(),
                        key="cmc_cal_end_date"
                    )
                    end_datetime = pd.to_datetime(end_date_selected) + pd.Timedelta(hours=23, minutes=59, seconds=59)

            # 2. Pilihan Metode Rentang Data Pelatihan
            cmc_method_options = [
                "Pilihan Cepat Rentang Waktu (1m - 30T)",
                "Gunakan Jumlah Hari / Jam / Menit Terakhir",
                "Rentang Slider (Tahun, Bulan, Hari, Jam, Menit)",
                "Pilih Tanggal dengan Kalender"
            ]
            selected_cmc_method = st.radio(
                "Pilihan Metode Memilih Data Pelatihan (CoinMarketCap):",
                options=cmc_method_options,
                index=0
            )

            if selected_cmc_method == "Pilihan Cepat Rentang Waktu (1m - 30T)":
                cmc_preset_keys = list(PRESET_DURATIONS.keys())
                
                # Baris Tombol Cepat (Warna khusus untuk 1H, 4H, 1D)
                st.markdown("<small><b>Pilih Cepat via Tombol (Khusus 1H 🟠, 4H 🔴, 1D 🟢):</b></small>", unsafe_allow_html=True)
                r_btn1 = st.columns(6)
                r_btn2 = st.columns(6)
                r_btn3 = st.columns(6)
                for idx_k, k_lbl in enumerate(cmc_preset_keys[:6]):
                    with r_btn1[idx_k]:
                        lbl_styled = format_quick_button_label(k_lbl)
                        b_type = "primary" if st.session_state.cmc_preset_selected == k_lbl else "secondary"
                        if st.button(lbl_styled, key=f"btn_p_cmc_{k_lbl}", type=b_type, use_container_width=True):
                            st.session_state.cmc_preset_selected = k_lbl
                            st.rerun()
                for idx_k, k_lbl in enumerate(cmc_preset_keys[6:12]):
                    with r_btn2[idx_k]:
                        lbl_styled = format_quick_button_label(k_lbl)
                        b_type = "primary" if st.session_state.cmc_preset_selected == k_lbl else "secondary"
                        if st.button(lbl_styled, key=f"btn_p_cmc_{k_lbl}", type=b_type, use_container_width=True):
                            st.session_state.cmc_preset_selected = k_lbl
                            st.rerun()
                for idx_k, k_lbl in enumerate(cmc_preset_keys[12:]):
                    with r_btn3[idx_k]:
                        lbl_styled = format_quick_button_label(k_lbl)
                        b_type = "primary" if st.session_state.cmc_preset_selected == k_lbl else "secondary"
                        if st.button(lbl_styled, key=f"btn_p_cmc_{k_lbl}", type=b_type, use_container_width=True):
                            st.session_state.cmc_preset_selected = k_lbl
                            st.rerun()

                active_cmc_preset = st.session_state.cmc_preset_selected
                st.info(f"⏱️ Rentang Waktu Pelatihan Terpilih: **{active_cmc_preset}**")
                total_duration = PRESET_DURATIONS[active_cmc_preset]
                duration_str = f"Preset {active_cmc_preset}"
                days = max(1, total_duration.days)

            elif selected_cmc_method == "Gunakan Jumlah Hari / Jam / Menit Terakhir":
                st.markdown("Masukkan jumlah hari, jam, atau menit terakhir yang diinginkan (tidak dibatasi):")
                c_num1, c_num2, c_num3 = st.columns(3)
                with c_num1:
                    input_days = st.number_input("📅 Jumlah Hari Terakhir:", min_value=0, value=30, step=1)
                with c_num2:
                    input_hours = st.number_input("⏰ Jumlah Jam Terakhir:", min_value=0, value=0, step=1)
                with c_num3:
                    input_mins = st.number_input("⏱️ Jumlah Menit Terakhir:", min_value=0, value=0, step=1)

                total_duration = timedelta(days=input_days, hours=input_hours, minutes=input_mins)
                if total_duration.total_seconds() < 1800:
                    total_duration = timedelta(minutes=30)
                duration_str = f"{input_days} Hari {input_hours} Jam {input_mins} Menit"
                days = max(1, total_duration.days)

            elif selected_cmc_method == "Rentang Slider (Tahun, Bulan, Hari, Jam, Menit)":
                st.markdown("Pilih rentang waktu pelatihan menggunakan slider:")
                c_sl1, c_sl2 = st.columns(2)
                with c_sl1:
                    years_ago = st.slider('📅 Tahun (0 - 30):', 0, 30, 30)
                    months_ago = st.slider('📅 Bulan Tambahan (0 - 11):', 0, 11, 0)
                    days_ago = st.slider('📅 Hari Tambahan (0 - 30):', 0, 30, 0)
                with c_sl2:
                    hours_ago = st.slider('⏰ Jam Tambahan (0 - 23):', 0, 23, 0)
                    mins_ago = st.slider('⏱️ Menit Tambahan (0 - 59):', 0, 59, 0)

                total_days = (years_ago * 365) + (months_ago * 30) + days_ago
                total_duration = timedelta(days=total_days, hours=hours_ago, minutes=mins_ago)
                if total_duration.total_seconds() < 1800:
                    total_duration = timedelta(minutes=30)
                duration_str = f"{years_ago} Tahun {months_ago} Bulan {days_ago} Hari {hours_ago} Jam {mins_ago} Menit"
                days = max(1, total_duration.days)

            else: # "Pilih Tanggal dengan Kalender"
                default_start = (end_datetime - timedelta(days=365*30)).date()
                c_cal1, c_cal2 = st.columns(2)
                with c_cal1:
                    cal_start = st.date_input(
                        "Tanggal Mulai Pelatihan:",
                        value=default_start,
                        min_value=date(2010, 1, 1),
                        max_value=end_datetime.date(),
                        key="cmc_custom_start_cal"
                    )
                with c_cal2:
                    cal_end = st.date_input(
                        "Tanggal Selesai Pelatihan:",
                        value=end_datetime.date(),
                        min_value=cal_start,
                        max_value=latest_avail_dt.date() if hasattr(latest_avail_dt, 'date') else date.today(),
                        key="cmc_custom_end_cal"
                    )
                start_datetime = pd.to_datetime(cal_start)
                end_datetime = pd.to_datetime(cal_end) + pd.Timedelta(hours=23, minutes=59, seconds=59)
                total_duration = end_datetime - start_datetime
                duration_str = f"Kalender ({cal_start} s/d {cal_end})"
                days = max(1, total_duration.days)

            # Ekstrak data berdasarkan start_datetime / total_duration dan end_datetime
            if not full_data.empty:
                sub_full = full_data[full_data.index <= end_datetime].copy()
                if sub_full.empty:
                    sub_full = full_data.copy()
                start_ts = end_datetime - total_duration
                data = sub_full[sub_full.index >= start_ts].copy()
                if len(data) < 30:
                    data = sub_full.tail(min(len(sub_full), 120)).copy()
            else:
                data = full_data.copy()

        else: # yFinance
            if 'yf_end_offset_btn' not in st.session_state:
                st.session_state.yf_end_offset_btn = None
            if 'yf_preset_selected' not in st.session_state:
                st.session_state.yf_preset_selected = "30 T"

            # 1. Pilihan Tanggal Selesai (Hari ini / Uncheck Kalender & Tombol)
            use_today_end = st.checkbox("Gunakan hingga tanggal terbaru (Hari ini)", value=True, key="chk_use_today_yf")
            latest_avail_dt = pd.to_datetime(full_data.index[-1]).date() if not full_data.empty else date.today()

            if use_today_end:
                end_date_obj = latest_avail_dt
                st.session_state.yf_end_offset_btn = None
            else:
                st.markdown("**Pilih Tanggal Selesai Pelatihan:**")
                st.markdown("<small><b>Tentukan Titik Akhir Berdasarkan Waktu Tersedia dari Tanggal Terakhir (Khusus 1D 🟢):</b></small>", unsafe_allow_html=True)
                
                yf_end_offsets = [
                    ("1 D", timedelta(days=1)),
                    ("1 W", timedelta(days=7)),
                    ("1 M", timedelta(days=30)),
                    ("90 D", timedelta(days=90)),
                    ("YTD", timedelta(days=365))
                ]
                btn_yf_cols = st.columns(5)
                for b_i, (t_lbl, t_delta) in enumerate(yf_end_offsets):
                    with btn_yf_cols[b_i]:
                        lbl_styled = format_quick_button_label(t_lbl)
                        b_type = "primary" if st.session_state.yf_end_offset_btn == t_lbl else "secondary"
                        if st.button(lbl_styled, key=f"btn_end_yf_{t_lbl}", type=b_type, use_container_width=True):
                            st.session_state.yf_end_offset_btn = t_lbl
                            st.rerun()

                selected_yf_offset = st.session_state.yf_end_offset_btn
                if selected_yf_offset:
                    offset_dict = dict(yf_end_offsets)
                    delta_val = offset_dict.get(selected_yf_offset, timedelta(0))
                    end_date_obj = latest_avail_dt - delta_val
                    st.info(f"📍 Tanggal Selesai Terpilih via Tombol: **{selected_yf_offset} yang lalu** (`{end_date_obj}`).")
                else:
                    end_date_obj = st.date_input(
                        "📅 Tanggal Selesai Pelatihan (Kalender):",
                        value=latest_avail_dt,
                        min_value=date(1990, 1, 1),
                        max_value=latest_avail_dt,
                        key="yf_cal_end_date"
                    )

            # 2. Pilihan Metode Rentang Data Pelatihan yFinance
            method_options = [
                "Pilihan Cepat Rentang Waktu (1D - 30T)",
                "Gunakan Jumlah Hari Terakhir",
                "Rentang Slider (Tahun, Bulan, Hari)",
                "Pilih Tanggal dengan Kalender"
            ]
            selected_method = st.radio(
                "Pilihan Metode Memilih Data Pelatihan:",
                options=method_options,
                index=0
            )

            yf_preset_keys = ["1 D", "1 W", "1 M", "90 D", "YTD", "2 T", "3 T", "4 T", "5 T", "10 T", "30 T"]

            if selected_method == "Pilihan Cepat Rentang Waktu (1D - 30T)":
                st.markdown("<small><b>Pilih Cepat via Tombol (Khusus 1D 🟢):</b></small>", unsafe_allow_html=True)
                r_btn_yf1 = st.columns(6)
                r_btn_yf2 = st.columns(5)
                for idx_k, k_lbl in enumerate(yf_preset_keys[:6]):
                    with r_btn_yf1[idx_k]:
                        lbl_styled = format_quick_button_label(k_lbl)
                        b_type = "primary" if st.session_state.yf_preset_selected == k_lbl else "secondary"
                        if st.button(lbl_styled, key=f"btn_p_yf_{k_lbl}", type=b_type, use_container_width=True):
                            st.session_state.yf_preset_selected = k_lbl
                            st.rerun()
                for idx_k, k_lbl in enumerate(yf_preset_keys[6:]):
                    with r_btn_yf2[idx_k]:
                        lbl_styled = format_quick_button_label(k_lbl)
                        b_type = "primary" if st.session_state.yf_preset_selected == k_lbl else "secondary"
                        if st.button(lbl_styled, key=f"btn_p_yf_{k_lbl}", type=b_type, use_container_width=True):
                            st.session_state.yf_preset_selected = k_lbl
                            st.rerun()

                active_yf_preset = st.session_state.yf_preset_selected
                st.info(f"⏱️ Rentang Waktu Pelatihan Terpilih: **{active_yf_preset}**")
                total_duration = PRESET_DURATIONS[active_yf_preset]
                days = max(120, total_duration.days)
                start_date_obj = end_date_obj - timedelta(days=days)

            elif selected_method == "Gunakan Jumlah Hari Terakhir":
                st.markdown("Masukkan jumlah hari pelatihan yang diinginkan (tidak dibatasi):")
                days = st.number_input("📅 Jumlah Hari Terakhir untuk Pelatihan:", min_value=120, value=365*30, step=1)
                start_date_obj = end_date_obj - timedelta(days=days)

            elif selected_method == "Rentang Slider (Tahun, Bulan, Hari)":
                st.markdown("Pilih rentang waktu pelatihan menggunakan slider:")
                c_yf1, c_yf2, c_yf3 = st.columns(3)
                with c_yf1:
                    years_ago = st.slider('📅 Tahun yang lalu (0 - 30):', 0, 30, 30)
                with c_yf2:
                    months_ago = st.slider('📅 Bulan tambahan (0 - 11):', 0, 11, 0)
                with c_yf3:
                    days_ago = st.slider('📅 Hari tambahan (0 - 30):', 0, 30, 0)
                
                days = (years_ago * 365) + (months_ago * 30) + days_ago
                if days < 120:
                    days = 120
                start_date_obj = end_date_obj - timedelta(days=days)

            else: # "Pilih Tanggal dengan Kalender"
                default_start = end_date_obj - timedelta(days=365*30)
                start_date_selected = st.date_input(
                    "Tanggal Mulai Pelatihan:",
                    value=default_start,
                    min_value=date(1990, 1, 1),
                    max_value=end_date_obj,
                    key="yf_cal_start_date"
                )
                start_date_obj = start_date_selected
                days = (end_date_obj - start_date_obj).days
                if days < 120:
                    days = 120

            with st.popover("Tips Memilih Data Pelatihan"):
                st.info('Ket: Semakin lama hari yang dipilih, maka jumlah hari Prediksi dapat dilakukan dengan lebih banyak, namun prediksi menjadi lebih tidak akurat.', icon=":material/notes:")
                st.warning('Ket: Secara Default menggunakan 30 Tahun yang lalu (atau data historis maksimal yang tersedia).', icon=":material/pan_tool_alt:")
                st.warning('Ket: Jika menggunakan Kalender, secara default tanggal mulai diset ke 30 Tahun yang lalu dan tanggal selesai diset ke tanggal terbaru.', icon=":material/calendar_month:")
                st.warning('Ket: Jumlah Minimal 4 Bulan atau 120 hari yang lalu, yang hanya dapat melakukan prediksi hingga 3 hari kedepan, Perhatikanlah pada bagian Pra-pemrosesan data "Ukuran data pengujian" jumlahnya sebanding dengan jumlah hari yang dapat anda lakukan untuk prediksi kedepan.', icon=":material/exclamation:")

            # Mengubah Format Tanggal data Pelatihan
            start_date = start_date_obj.strftime("%Y-%m-%d")
            end_date = end_date_obj.strftime("%Y-%m-%d")

            @st.cache_data
            def load_training_data(stock, start_date, end_date):
                return load_data(stock, start_date, end_date)

            try:
                data = load_training_data(stock, start_date, end_date).copy()
            except Exception as e:
                data = load_training_data(stock, start_date, end_date).copy()

            if not data.empty:
                actual_days = (data.index[-1] - data.index[0]).days
                y_cnt = actual_days // 365
                rem_days = actual_days % 365
                m_cnt = rem_days // 30
                d_cnt = rem_days % 30
                duration_str = f"{y_cnt} Tahun {m_cnt} Bulan {d_cnt} Hari"
            else:
                actual_days = 0
                duration_str = "0 Tahun 0 Bulan 0 Hari"

        st.subheader("Data Pelatihan yang telah dipilih")
        if is_cmc:
            st.write(f"Rentang Waktu Terpilih: **{duration_str}** ({len(data)} baris data).")
        else:
            st.write(f"Jumlah Hari yang dipilih **{actual_days}** ({duration_str}).")

        st.write("Mulai")
        st.write(format_df_for_display(data.head(1)))
        st.write("Hingga")
        st.write(format_df_for_display(data.tail(1)))

        if not data.empty and len(data) >= 2:
            c_start_t = safe_float(data['Close'].iloc[0])
            c_end_t = safe_float(data['Close'].iloc[-1])
            tot_chg_t = ((c_end_t - c_start_t) / c_start_t) * 100.0 if c_start_t > 0 else 0.0
            tot_sign_t = f":green[▲ +{tot_chg_t:,.2f}%]" if tot_chg_t >= 0 else f":red[▼ {tot_chg_t:,.2f}%]"
            tot_diff_t = c_end_t - c_start_t
            diff_sign_t = "+" if tot_diff_t >= 0 else ""
            tot_diff_str_t = smart_format(tot_diff_t, prefix=curr_prefix)
            st.markdown(f"**Performa Perubahan Data Pelatihan:** {tot_sign_t} (`{diff_sign_t}{tot_diff_str_t}`)")

        # Plot Interaktif dengan Plotly untuk data pelatihan (Garis Kuning)
        fig_train = plot_interactive_history(data, f'Data Pelatihan Harga {asset_type}', f'Harga {asset_type}', '#D6C36B', curr_prefix=curr_prefix)
        if fig_train is not None:
            render_plotly_with_tools(fig_train, key="fig_train_data_history")

        with st.expander(f"📈 Grafik Tren Riwayat Harga Candlestick, VPVR, Volume, ATR & Delta Volume (Data Pelatihan yang Dipilih)", expanded=False):
            fig_train_comp = plot_comprehensive_market_indicators(data, f'Indikator Pasar Komprehensif Candlestick & VPVR (Data Pelatihan {asset_type})', curr_prefix=curr_prefix, asset_type=asset_type)
            if fig_train_comp is not None:
                render_plotly_with_tools(fig_train_comp, key="fig_train_data_comp_indicators")

        # Toggle Grafik Mini Tren Riwayat Harga, Volume, ATR & Delta
        expander_title = "📈 Grafik Mini Tren Riwayat Harga, Volume, ATR & Delta Volume (1m, 3m, 5m, 15m, 30m, 1H, 2H, 4H, 12H, 1D)" if is_cmc else "📈 Grafik Mini Tren Riwayat Harga, Volume, ATR & Delta Volume (1D, 1W, 1M, 90D, YTD)"
        with st.expander(expander_title, expanded=False):
            if not data.empty:
                if is_cmc:
                    period_slices = [
                        ("1m", get_time_period_slice(data, timedelta(minutes=1)), get_time_change_pct(data, timedelta(minutes=1))),
                        ("3m", get_time_period_slice(data, timedelta(minutes=3)), get_time_change_pct(data, timedelta(minutes=3))),
                        ("5m", get_time_period_slice(data, timedelta(minutes=5)), get_time_change_pct(data, timedelta(minutes=5))),
                        ("15m", get_time_period_slice(data, timedelta(minutes=15)), get_time_change_pct(data, timedelta(minutes=15))),
                        ("30m", get_time_period_slice(data, timedelta(minutes=30)), get_time_change_pct(data, timedelta(minutes=30))),
                        ("1H", get_time_period_slice(data, timedelta(hours=1)), get_time_change_pct(data, timedelta(hours=1))),
                        ("2H", get_time_period_slice(data, timedelta(hours=2)), get_time_change_pct(data, timedelta(hours=2))),
                        ("4H", get_time_period_slice(data, timedelta(hours=4)), get_time_change_pct(data, timedelta(hours=4))),
                        ("12H", get_time_period_slice(data, timedelta(hours=12)), get_time_change_pct(data, timedelta(hours=12))),
                        ("1D", get_time_period_slice(data, timedelta(days=1)), get_time_change_pct(data, timedelta(days=1)))
                    ]
                else:
                    period_slices = [
                        ("1 Hari (1D)", get_period_slice(data, 1), get_change_pct(data, 1)),
                        ("1 Minggu (1W)", get_period_slice(data, 7), get_change_pct(data, 7)),
                        ("1 Bulan (1M)", get_period_slice(data, 30), get_change_pct(data, 30)),
                        ("90 Hari (90D)", get_period_slice(data, 90), get_change_pct(data, 90)),
                        ("1 Tahun (YTD)", get_period_slice(data, 365), get_change_pct(data, 365))
                    ]
                
                # 1. Baris Chart Harga (Garis Close Tanpa VWAP)
                st.markdown(f"**1. Grafik Mini Tren Harga {asset_type} (Garis Hijau: Tren Naik / Garis Merah: Tren Turun):**")
                cols_price = st.columns(min(len(period_slices), 5))
                for idx, (label, s_df, chg) in enumerate(period_slices):
                    col = cols_price[idx % 5]
                    with col:
                        if chg is not None and not s_df.empty and len(s_df) >= 2:
                            c_arr = extract_1d_array(s_df['Close'] if 'Close' in s_df.columns else s_df.iloc[:, 0])
                            mean_c = np.mean(c_arr) if len(c_arr) > 0 else 0.0
                            mean_c_str = smart_format(mean_c, prefix=curr_prefix)
                            badge_sign = f":green[▲ +{chg:.2f}%]" if chg >= 0 else f":red[▼ {chg:.2f}%]"
                            st.markdown(f"<small><b>{label}</b> ({badge_sign})<br>Rata2: <code>{mean_c_str}</code></small>", unsafe_allow_html=True)
                            is_pos = (chg >= 0)
                            fig = render_sparkline_chart(s_df['Close'] if 'Close' in s_df.columns else s_df.iloc[:, 0], is_positive=is_pos, chart_type='line', fill=True)
                            st.pyplot(fig)
                            plt.close(fig)
                        else:
                            st.markdown(f"<small><b>{label}</b> (-)<br>Rata2: <code>-</code></small>", unsafe_allow_html=True)
                            st.write("-")
                            
                # 2. Baris Chart Volume Transaksi Disatukan dengan Garis ATR (Kuning)
                st.markdown(f"**2. Grafik Mini Volume Transaksi & ATR (Bar Hijau/Merah: Volume, Garis Kuning: ATR Volatilitas):**")
                cols_vol_atr = st.columns(min(len(period_slices), 5))
                for idx, (label, s_df, chg) in enumerate(period_slices):
                    col = cols_vol_atr[idx % 5]
                    with col:
                        if chg is not None and not s_df.empty and len(s_df) >= 2:
                            vol_arr = extract_1d_array(s_df['Volume']) if 'Volume' in s_df.columns else np.array([])
                            if len(vol_arr) >= 2 and vol_arr[0] > 0:
                                v_chg = ((vol_arr[-1] - vol_arr[0]) / vol_arr[0]) * 100.0
                                v_badge = f":green[▲ +{v_chg:.1f}%]" if v_chg >= 0 else f":red[▼ {v_chg:.1f}%]"
                            else:
                                v_badge = "-"
                            
                            atr_arr = calculate_atr_series(s_df)
                            latest_atr_str = smart_format(atr_arr[-1], prefix=curr_prefix) if len(atr_arr) > 0 else "-"
                            
                            st.markdown(f"<small><b>Vol {label}</b> ({v_badge})<br>ATR: <span style='color:#FFB300;'><b>{latest_atr_str}</b></span></small>", unsafe_allow_html=True)
                            if not s_df.empty and 'Volume' in s_df.columns:
                                bar_cols = get_bar_colors_for_volume(s_df)
                                fig = render_combined_volume_atr_sparkline(s_df['Volume'], atr_arr, bar_colors=bar_cols)
                                st.pyplot(fig)
                                plt.close(fig)
                            else:
                                st.write("-")
                        else:
                            st.markdown(f"<small><b>Vol {label}</b> (-)<br>ATR: -</small>", unsafe_allow_html=True)
                            st.write("-")

                # 3. Baris Chart Delta Volume Harian (Bar Hijau Net Buy & Merah Net Sell)
                st.markdown(f"**3. Grafik Mini Delta Volume (Bar Hijau: Net Buy, Bar Merah: Net Sell):**")
                cols_delta = st.columns(min(len(period_slices), 5))
                for idx, (label, s_df, chg) in enumerate(period_slices):
                    col = cols_delta[idx % 5]
                    with col:
                        if chg is not None and not s_df.empty and len(s_df) >= 2:
                            daily_delta, daily_delta_cols = calculate_daily_delta_volume_series(s_df)
                            if len(daily_delta) > 0:
                                net_delta = np.sum(daily_delta)
                                d_pos = net_delta >= 0
                                d_badge = f":green[▲ +{format_market_cap(net_delta)} Net Buy]" if d_pos else f":red[▼ -{format_market_cap(abs(net_delta))} Net Sell]"
                            else:
                                d_pos = True
                                d_badge = "-"
                            st.markdown(f"<small><b>Delta {label}</b><br>{d_badge}</small>", unsafe_allow_html=True)
                            if len(daily_delta) > 0:
                                fig = render_sparkline_chart(daily_delta, is_positive=d_pos, chart_type='bar', bar_colors=daily_delta_cols)
                                st.pyplot(fig)
                                plt.close(fig)
                            else:
                                st.write("-")
                        else:
                            st.markdown(f"<small><b>Delta {label}</b><br>-</small>", unsafe_allow_html=True)
                            st.write("-")

        with st.popover("Tampilkan Semua Data Pelatihan"):
            st.write(format_df_for_display(data))
            csv_train = data.to_csv().encode('utf-8')
            st.download_button(
                label="📥 Unduh Data Pelatihan sebagai CSV",
                data=csv_train,
                file_name=f"data_pelatihan_{stock}.csv",
                mime="text/csv",
                key="btn_dl_train_data"
            )

    with st.expander("3. Pra-pemrosesan Data"):

        is_valid_data = len(data) >= 30 if is_cmc else (days >= 120)

        if is_valid_data:

            with st.popover("⚙️ Pengaturan Panjang Sekuens (Lookback)"):
                if is_cmc:
                    all_seq_options = [3, 5, 10, 15, 20, 30, 45, 60, 90, 120, 180]
                    max_seq = max(3, min(180, len(data) // 3))
                    valid_seq_options = [s for s in all_seq_options if s <= max_seq]
                    if not valid_seq_options:
                        valid_seq_options = [max(1, len(data) // 4)]
                    default_seq = 15 if 15 in valid_seq_options else (30 if 30 in valid_seq_options else valid_seq_options[-1])
                    seq_length = st.select_slider(
                        "Panjang Sekuens (Lookback Window) [Langkah Data]",
                        options=valid_seq_options,
                        value=default_seq,
                        key=f"seq_slider_cmc_{len(data)}"
                    )
                    st.info(f'Rekomendasi Default: **{default_seq} Langkah Data** (Disesuaikan otomatis sesuai ukuran data: {len(data)} baris).', icon=":material/recommend:")
                else:
                    all_seq_options = [5, 10, 15, 20, 30, 45, 60, 90, 120, 180]
                    max_seq = max(5, min(180, len(data) // 3))
                    valid_seq_options = [s for s in all_seq_options if s <= max_seq]
                    if not valid_seq_options:
                        valid_seq_options = [30]
                    default_seq = 60 if 60 in valid_seq_options else (30 if 30 in valid_seq_options else valid_seq_options[-1])
                    seq_length = st.select_slider(
                        "Panjang Sekuens (Hari)",
                        options=valid_seq_options,
                        value=default_seq,
                        key=f"seq_slider_yf_{len(data)}"
                    )
                    st.info(f'Rekomendasi Default: **{default_seq} Hari** (Disesuaikan otomatis sesuai ukuran data: {len(data)} hari).', icon=":material/recommend:")
                st.warning('Ket: Sekuens terlalu pendek kehilangan konteks tren, sekuens terlalu panjang menambah dimensi & mengurangi jumlah sampel data.', icon=":material/timeline:")

            with st.popover("⚙️ Pengaturan Normalisasi (MinMaxScaler)"):
                scaler_option = st.radio("Rentang Normalisasi:", options=["(0, 1)", "(-1, 1)"], index=0)
                feature_range = (0, 1) if scaler_option == "(0, 1)" else (-1, 1)
                st.info('Rekomendasi Default: **(0, 1)**. Sangat optimal untuk aktivasi ReLU dan harga aset non-negatif.', icon=":material/recommend:")
                st.warning('Ket: Rentang (-1, 1) dapat digunakan jika menggunakan aktivasi simetris tanh di seluruh model.', icon=":material/tune:")

            with st.popover("⚙️ Pengaturan Pembagian Data (Train/Test Split)"):
                split_pct = st.select_slider("Persentase Data Pelatihan (%)", options=[50, 60, 70, 75, 80, 85, 90], value=80)
                train_ratio = split_pct / 100.0
                st.info('Rekomendasi Default: **80% Pelatihan : 20% Pengujian** (Rentang Paling Akurat: **75%–85%**).', icon=":material/recommend:")
                st.warning('Ket: Porsi pelatihan <70% membuat model kurang belajar, porsi >85% membuat data pengujian terlalu sedikit untuk validasi.', icon=":material/pie_chart:")

            def preprocess_data(data, seq_length, feature_range, train_ratio):
                scaler = MinMaxScaler(feature_range=feature_range)
                scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

                X, y = [], []
                for i in range(seq_length, len(scaled_data)):
                    X.append(scaled_data[i-seq_length:i, 0])
                    y.append(scaled_data[i, 0])

                X, y = np.array(X), np.array(y)

                split = int(train_ratio * len(X))
                x_train, x_test = X[:split], X[split:]
                y_train, y_test = y[:split], y[split:]

                x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
                x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

                return x_train, x_test, y_train, y_test, scaler

            x_train, x_test, y_train, y_test, scaler = preprocess_data(data, seq_length, feature_range, train_ratio)

            total_samples = x_train.shape[0] + x_test.shape[0]
            train_percentage = (x_train.shape[0] / total_samples) * 100 if total_samples > 0 else 0
            test_percentage = (x_test.shape[0] / total_samples) * 100 if total_samples > 0 else 0

            with st.popover("Detail Pra-pemrosesan Data"):
                st.info(f'Ukuran Panjang Sekuens (`seq_length`): Menggunakan **{seq_length}** langkah historis untuk memprediksi harga pada langkah berikutnya.', icon=":material/timeline:")
                st.warning(f'Pembagian Data: Data di-scaling dengan `MinMaxScaler{feature_range}` lalu dibagi menjadi **{train_percentage:.1f}%** data pelatihan ({x_train.shape[0]} sampel) dan **{test_percentage:.1f}%** data pengujian ({x_test.shape[0]} sampel).', icon=":material/pie_chart:")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Ukuran data pelatihan", f"{x_train.shape[0]} sampel")
                st.metric("Persentase data pelatihan", f"{train_percentage:.2f}%")
            with col2:
                st.metric("Ukuran data pengujian", f"{x_test.shape[0]} sampel")
                st.metric("Persentase data pengujian", f"{test_percentage:.2f}%")

            st.success("Pra-pemrosesan Data selesai!")
        else:
            if is_cmc:
                st.warning('Harus Memilih Rentang Waktu Minimal 30 Menit atau 30 Baris Data', icon=":material/exclamation:")
            else:
                st.warning('Harus Memilih Jumlah Hari Minimal 4 Bulan atau 120 hari', icon=":material/exclamation:")

    with st.expander("4. Perancangan Model CNN-GRU"):

        if is_valid_data:

            st.subheader("Arsitektur Model & Pengaturan Hyperparameter:")

            with st.popover("⚙️ Pengaturan Lapisan Conv1D"):
                conv_filters = st.select_slider("Jumlah Filter Conv1D", options=[16, 32, 48, 64, 96, 128, 256], value=64)
                kernel_size = st.select_slider("Ukuran Kernel (Kernel Size)", options=[2, 3, 4, 5, 7], value=3)
                conv_activation = st.selectbox("Fungsi Aktivasi Conv1D", options=["relu", "tanh", "elu", "linear"], index=0)
                st.info('Rekomendasi Default: **64 filter**, **kernel 3**, aktivasi **ReLU** (Rentang Akurat: 32–64 filter, kernel 3–5).', icon=":material/recommend:")
                st.warning('Ket: Filter Conv1D mengekstrak fitur spasial & momentum lokal jangka pendek dari sekuens harga.', icon=":material/layers:")

            with st.popover("⚙️ Pengaturan Lapisan GRU"):
                gru_units_1 = st.select_slider("Unit GRU Layer 1", options=[16, 32, 50, 64, 96, 128, 256], value=50)
                gru_units_2 = st.select_slider("Unit GRU Layer 2", options=[16, 32, 50, 64, 96, 128, 256], value=50)
                st.info('Rekomendasi Default: **50 Unit** per layer (Rentang Akurat: **32–64 Unit**).', icon=":material/recommend:")
                st.warning('Ket: GRU menangkap ketergantungan temporal jangka panjang. Nilai di atas 128 meningkatkan risiko overfitting pada deret waktu.', icon=":material/memory:")

            with st.popover("⚙️ Pengaturan Regularisasi Dropout"):
                dropout_rate = st.select_slider("Tingkat Dropout (Dropout Rate)", options=[0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5], value=0.2)
                st.info('Rekomendasi Default: **0.2 (20%)** (Rentang Akurat: **0.1–0.3**).', icon=":material/recommend:")
                st.warning('Ket: Dropout 0.2 mencegah ko-adaptasi neuron yang menyebabkan overfitting tanpa membuang representasi penting.', icon=":material/security:")

            def create_model(seq_len, c_filters, k_size, c_act, g_u1, g_u2, d_rate, lr=0.001):
                model = Sequential([
                    Conv1D(filters=c_filters,
                        kernel_size=k_size,
                        activation=c_act,
                        input_shape=(seq_len, 1)),
                    GRU(g_u1, return_sequences=True),
                    Dropout(d_rate),
                    GRU(g_u2),
                    Dense(1)
                ])

                try:
                    model.build(input_shape=(None, seq_len, 1))
                except Exception:
                    pass

                model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
                return model

            preview_model = create_model(seq_length, conv_filters, kernel_size, conv_activation, gru_units_1, gru_units_2, dropout_rate)
            try:
                total_params = preview_model.count_params()
            except Exception:
                total_params = 0

            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.metric("Total Parameter Model (Trainable)", f"{total_params:,}")
            with col_p2:
                st.metric("Jumlah Lapisan (Layers)", f"{len(preview_model.layers)} Lapisan")

            with st.popover("Detail Arsitektur & Ringkasan Parameter"):
                st.info(f'Lapisan Ekstraksi Fitur: `Conv1D` ({conv_filters} filter, kernel {kernel_size}, aktivasi {conv_activation}) untuk mengekstrak pola fitur spasial dari sekuens {seq_length} langkah.', icon=":material/layers:")
                st.warning(f'Lapisan Memori & Regulasi: `GRU` ({gru_units_1} unit, seq) $\\rightarrow$ `Dropout` ({dropout_rate}) $\\rightarrow$ `GRU` ({gru_units_2} unit) untuk menangkap pola temporal sekuensial.', icon=":material/memory:")
                st.warning('Lapisan Output & Kompilasi: Lapisan `Dense(1)` dikompilasi dengan optimizer `Adam` dan loss function `MSE`.', icon=":material/check_circle:")
                
                layer_data = []
                for lyr in preview_model.layers:
                    shape_str = "-"
                    try:
                        if hasattr(lyr, 'output_shape') and lyr.output_shape is not None:
                            shape_str = str(lyr.output_shape)
                        elif hasattr(lyr, 'compute_output_shape'):
                            shape_str = str(lyr.compute_output_shape((None, seq_length, 1)))
                    except Exception:
                        shape_str = "-"
                    
                    try:
                        param_cnt = f"{lyr.count_params():,}"
                    except Exception:
                        param_cnt = "-"

                    layer_data.append({
                        "Nama Lapisan": lyr.name,
                        "Tipe Lapisan": lyr.__class__.__name__,
                        "Bentuk Output": shape_str,
                        "Jumlah Parameter": param_cnt
                    })
                st.dataframe(pd.DataFrame(layer_data), use_container_width=True)

            def get_model(seq_len, lr=0.001):
                return create_model(seq_len, conv_filters, kernel_size, conv_activation, gru_units_1, gru_units_2, dropout_rate, lr=lr)

            st.success("Perancangan Model CNN-GRU selesai!")

        else:
            if is_cmc:
                st.warning('Harus Memilih Rentang Waktu Minimal 30 Menit atau 30 Baris Data', icon=":material/exclamation:")
            else:
                st.warning('Harus Memilih Jumlah Hari Minimal 4 Bulan atau 120 hari', icon=":material/exclamation:")

    with st.expander("5. Pelatihan Model", True):

        if 'btn_check' not in st.session_state:
            st.session_state.btn_check = 0

        btn_check = 0

        if is_valid_data:

            # Hyperparameter Tuning popover
            with st.popover("⚙️ Pengaturan Epoch, Batch & Learning Rate"):
                is_crypto = is_cmc or is_crypto_ticker(stock)
                
                if is_crypto:
                    default_epoch = 100
                    default_batch = 4
                    default_lr = 0.0005
                    acc_epoch_desc = "30–120 (Optimal Crypto: 80–120)"
                    acc_batch_desc = "4–16 (Optimal Crypto: 4–8)"
                    acc_lr_desc = "0.0001–0.001 (Optimal Crypto: 0.0003–0.0008)"
                else:
                    default_epoch = 50
                    default_batch = 16
                    default_lr = 0.001
                    acc_epoch_desc = "30–100 (Optimal Saham: 40–60)"
                    acc_batch_desc = "8–32 (Optimal Saham: 16–32)"
                    acc_lr_desc = "0.0005–0.002 (Optimal Saham: 0.001)"

                epochs = st.slider("Jumlah Epoch:", min_value=10, max_value=200, value=default_epoch, step=5)
                batch_size = st.select_slider("Ukuran Batch (Batch Size):", options=[1, 2, 4, 8, 16, 32, 64, 128], value=default_batch)
                
                lr_options = [0.0001, 0.0003, 0.0005, 0.0008, 0.001, 0.002, 0.003, 0.005, 0.01]
                lr_index = lr_options.index(default_lr) if default_lr in lr_options else 4
                learning_rate = st.select_slider("Learning Rate (Adam Optimizer):", options=lr_options, value=lr_options[lr_index], format_func=lambda x: f"{x:.4f}")

                st.info(f"**Rekomendasi Parameter Optimal ({asset_type}):**\n"
                        f"- Epoch Default: **{default_epoch}** (Rentang Akurat: **{acc_epoch_desc}**)\n"
                        f"- Batch Size Default: **{default_batch}** (Rentang Akurat: **{acc_batch_desc}**)\n"
                        f"- Learning Rate Default: **{default_lr}** (Rentang Akurat: **{acc_lr_desc}**)", icon=":material/recommend:")
                st.warning("Ket: Batch size kecil (4-8) sangat efektif untuk menangkap volatilitas kripto, sedangkan batch moderat (16-32) lebih stabil untuk saham konvensional.", icon=":material/insights:")

            # Forecasting Options
            def get_cmc_forecast_options():
                return [
                    ("1 Menit (1m)", 1),
                    ("3 Menit (3m)", 3),
                    ("5 Menit (5m)", 5),
                    ("10 Menit (10m)", 10),
                    ("15 Menit (15m)", 15),
                    ("30 Menit (30m)", 30),
                    ("45 Menit (45m)", 45),
                    ("1 Jam (1h)", 60),
                    ("2 Jam (2h)", 120),
                    ("3 Jam (3h)", 180),
                    ("4 Jam (4h)", 240),
                    ("6 Jam (6h)", 360),
                    ("8 Jam (8h)", 480),
                    ("12 Jam (12h)", 720),
                    ("1 Hari (24h)", 1440),
                    ("2 Hari (48h)", 2880),
                    ("3 Hari (72h)", 4320),
                    ("1 Minggu (7D)", 10080)
                ]

            default_cmc_options_map = {
                1: ["1 Menit (1m)"],
                3: ["1 Menit (1m)", "3 Menit (3m)"],
                5: ["1 Menit (1m)", "3 Menit (3m)", "5 Menit (5m)"],
                10: ["1 Menit (1m)", "5 Menit (5m)", "10 Menit (10m)"],
                15: ["1 Menit (1m)", "5 Menit (5m)", "15 Menit (15m)"],
                30: ["1 Menit (1m)", "5 Menit (5m)", "15 Menit (15m)", "30 Menit (30m)"],
                45: ["5 Menit (5m)", "15 Menit (15m)", "30 Menit (30m)", "45 Menit (45m)"],
                60: ["5 Menit (5m)", "15 Menit (15m)", "30 Menit (30m)", "1 Jam (1h)"],
                120: ["15 Menit (15m)", "30 Menit (30m)", "1 Jam (1h)", "2 Jam (2h)"],
                180: ["15 Menit (15m)", "30 Menit (30m)", "1 Jam (1h)", "3 Jam (3h)"],
                240: ["30 Menit (30m)", "1 Jam (1h)", "2 Jam (2h)", "4 Jam (4h)"],
                360: ["30 Menit (30m)", "1 Jam (1h)", "3 Jam (3h)", "6 Jam (6h)"],
                480: ["1 Jam (1h)", "2 Jam (2h)", "4 Jam (4h)", "8 Jam (8h)"],
                720: ["1 Jam (1h)", "3 Jam (3h)", "6 Jam (6h)", "12 Jam (12h)"],
                1440: ["1 Jam (1h)", "6 Jam (6h)", "12 Jam (12h)", "1 Hari (24h)"],
                2880: ["6 Jam (6h)", "12 Jam (12h)", "1 Hari (24h)", "2 Hari (48h)"],
                4320: ["12 Jam (12h)", "1 Hari (24h)", "2 Hari (48h)", "3 Hari (72h)"],
                10080: ["1 Hari (24h)", "2 Hari (48h)", "3 Hari (72h)", "1 Minggu (7D)"]
            }

            def initialize_cmc_forecast_options(x_test):
                f_opts = get_cmc_forecast_options()
                f_dict = {name: steps for name, steps in f_opts}
                f_steps = x_test.shape[0]

                valid_opts = {name: steps for name, steps in f_dict.items() if steps <= f_steps}
                if not valid_opts:
                    max_s = max((steps for steps in f_dict.values() if steps <= f_steps), default=1)
                    valid_opts = {name: steps for name, steps in f_dict.items() if steps == max_s}
                if not valid_opts:
                    valid_opts = {"1 Menit (1m)": 1}

                closest_key = min(default_cmc_options_map.keys(), key=lambda x: abs(x - f_steps))
                def_opts = default_cmc_options_map[closest_key]
                def_opts = [o for o in def_opts if o in valid_opts]
                if not def_opts:
                    def_opts = list(valid_opts.keys())[:min(len(valid_opts), 4)]
                return valid_opts, def_opts

            def get_forecast_options(stock):
                return [
                    ("1 Hari", 1), ("2 Hari", 2), ("3 Hari", 3), ("4 Hari", 4), ("5 Hari", 5), ("6 Hari", 6),
                    ("1 Minggu", 7), ("2 Minggu", 14), ("3 Minggu", 21), ("1 Bulan", 30), ("2 Bulan", 60),
                    ("3 Bulan", 90), ("4 Bulan", 120), ("5 Bulan", 150), ("6 Bulan", 180), ("7 Bulan", 210),
                    ("8 Bulan", 240), ("9 Bulan", 270), ("10 Bulan", 300), ("11 Bulan", 330), ("1 Tahun", 365),
                    ("2 Tahun", 730)
                ]

            default_options_map = {
                3: ["1 Hari", "2 Hari", "3 Hari"],
                4: ["2 Hari", "3 Hari", "4 Hari"],
                5: ["3 Hari", "4 Hari", "5 Hari"],
                6: ["4 Hari", "5 Hari", "6 Hari"],
                7: ["5 Hari", "6 Hari", "1 Minggu"],
                14: ["6 Hari", "1 Minggu", "2 Minggu"],
                21: ["1 Minggu", "2 Minggu", "3 Minggu"],
                30: ["1 Minggu", "2 Minggu", "1 Bulan"],
                60: ["1 Minggu", "1 Bulan", "2 Bulan"],
                90: ["1 Minggu", "1 Bulan", "3 Bulan"],
                120: ["1 Minggu", "1 Bulan", "3 Bulan", "4 Bulan"],
                150: ["1 Minggu", "1 Bulan", "3 Bulan", "5 Bulan"],
                180: ["1 Minggu", "1 Bulan", "3 Bulan", "6 Bulan"],
                210: ["1 Minggu", "1 Bulan", "3 Bulan", "6 Bulan", "7 Bulan"],
                240: ["1 Minggu", "1 Bulan", "3 Bulan", "6 Bulan", "8 Bulan"],
                270: ["1 Minggu", "1 Bulan", "3 Bulan", "6 Bulan", "9 Bulan"],
                300: ["1 Minggu", "1 Bulan", "3 Bulan", "6 Bulan", "10 Bulan"],
                330: ["1 Minggu", "1 Bulan", "3 Bulan", "6 Bulan", "11 Bulan"],
                365: ["1 Minggu", "1 Bulan", "3 Bulan", "6 Bulan", "1 Tahun"],
                730: ["1 Minggu", "1 Bulan", "3 Bulan", "6 Bulan", "1 Tahun", "2 Tahun"]
            }

            def initialize_forecast_options(stock, x_test):
                f_opts = get_forecast_options(stock)
                f_dict = {name: d for name, d in f_opts}
                f_days = x_test.shape[0]
                valid_opts = {name: d for name, d in f_dict.items() if d <= f_days}
                if not valid_opts:
                    max_d = max(d for d in f_dict.values() if d <= f_days)
                    valid_opts = {name: d for name, d in f_dict.items() if d == max_d}
                closest_key = min(default_options_map.keys(), key=lambda x: abs(x - f_days))
                def_opts = default_options_map[closest_key]
                def_opts = [o for o in def_opts if o in valid_opts]
                return valid_opts, def_opts

            if is_cmc:
                forecast_options_dict, default_options = initialize_cmc_forecast_options(x_test)
            else:
                forecast_options_dict, default_options = initialize_forecast_options(stock, x_test)

            fc_key = f"selected_forecast_periods_{stock}_{len(x_test)}_{data_source}"
            if fc_key not in st.session_state:
                st.session_state[fc_key] = default_options

            st.write("**Pilihan Periode Forecasting:**")
            col_b1, col_b2, col_b3 = st.columns([1.2, 1.2, 1.2])
            with col_b1:
                if st.button("🔄 Pilih Default", help="Kembalikan ke pilihan periode default"):
                    st.session_state[fc_key] = default_options
                    st.rerun()
            with col_b2:
                if st.button("✅ Pilih Semua", help="Pilih semua periode yang tersedia"):
                    st.session_state[fc_key] = list(forecast_options_dict.keys())
                    st.rerun()
            with col_b3:
                if st.button("❌ Hapus Semua", help="Kosongkan pilihan periode"):
                    st.session_state[fc_key] = []
                    st.rerun()

            selected_periods = st.multiselect(
                "Pilih Periode Forecasting:",
                options=list(forecast_options_dict.keys()),
                key=fc_key
            )

            # Training function
            def train_model(x_train, y_train, epochs, batch_size, lr_val, _on_epoch_end):
                with st.spinner('Sedang Melatih model... Harap tunggu.'):
                    try:
                        model = get_model(x_train.shape[1], lr=lr_val)
                        history = model.fit(
                            x_train, y_train,
                            epochs=epochs,
                            batch_size=batch_size,
                            validation_split=0.1,
                            verbose=0,
                            callbacks=[LambdaCallback(on_epoch_end=_on_epoch_end)]
                        )
                        return model, history
                    except Exception as e:
                        st.error(f"Terjadi error saat melatih model: {str(e)}")
                        return None, None

            if st.button("Latih Model", type="primary"):
                start_time = time.time()
                progress_bar = st.progress(0)
                status_text = st.empty()
                time_estimate = st.empty()

                def on_epoch_end(epoch, logs):
                    progress = (epoch + 1) / epochs
                    progress_bar.progress(progress)
                    elapsed_time = time.time() - start_time
                    estimated_total_time = elapsed_time / progress if progress > 0 else 0
                    remaining_time = estimated_total_time - elapsed_time
                    status_text.text(f"Epoch {epoch + 1}/{epochs} - Loss: {logs.get('loss', 0):.4f}")
                    time_estimate.text(f"Estimasi waktu tersisa: {remaining_time:.2f} detik")

                model, history = train_model(x_train, y_train, epochs, batch_size, learning_rate, on_epoch_end)

                if model is not None:
                    end_time = time.time()
                    st.success(f"Pelatihan model selesai dalam {end_time - start_time:.2f} detik!")
                    st.session_state.training_completed = True
                    btn_check = 1

        else:
            if is_cmc:
                st.warning('Harus Memilih Rentang Waktu Minimal 30 Menit atau 30 Baris Data', icon=":material/exclamation:")
            else:
                st.warning('Harus Memilih Jumlah Hari Minimal 4 Bulan atau 120 hari', icon=":material/exclamation:")

    with st.expander("6. Evaluasi Model"):

        if btn_check == 1:

            start_time = time.time()
            progress_bar = st.progress(0)
            status_text = st.empty()
            time_estimate = st.empty()

            with st.spinner('Sedang Melakukan Evaluasi Model... Harap tunggu.'):

                predictions = model.predict(x_test, verbose=0)
                y_pred = scaler.inverse_transform(predictions)
                y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))

                perf_eval = evaluate_model_performance(y_test_original, y_pred) or {}

                st.subheader("Hasil Evaluasi:")
                e_col1, e_col2, e_col3 = st.columns(3)
                with e_col1:
                    render_colored_metric_card("Akurasi Model", f"{perf_eval.get('accuracy', 0):.3f}%", perf_eval.get('cat_acc', 'neutral'), "Tingkat ketepatan arah & nilai")
                with e_col2:
                    r2_display = f"{perf_eval.get('r2'):.3f}" if perf_eval.get('r2') is not None else "-"
                    render_colored_metric_card("R2 Score", r2_display, perf_eval.get('cat_r2', 'neutral'), "Proporsi varians terjelaskan")
                with e_col3:
                    render_colored_metric_card("MAPE", f"{perf_eval.get('mape', 0):.3f}", perf_eval.get('cat_mape', 'neutral'), "Rata-rata persentase galat")

                e_col4, e_col5, e_col6 = st.columns(3)
                with e_col4:
                    render_colored_metric_card("RMSE", smart_format(perf_eval.get('rmse', 0), default_decimals=3), perf_eval.get('cat_rmse', 'neutral'), "Akar kuadrat galat rata-rata")
                with e_col5:
                    render_colored_metric_card("MAE", smart_format(perf_eval.get('mae', 0), default_decimals=3), perf_eval.get('cat_mae', 'neutral'), "Rata-rata selisih absolut")
                with e_col6:
                    render_colored_metric_card("MSE", smart_format(perf_eval.get('mse', 0), default_decimals=3), perf_eval.get('cat_mse', 'neutral'), "Rata-rata kuadrat galat")

                comp_score = perf_eval.get('composite_score', 0)
                perf_label = perf_eval.get('label', 'Performa')
                perf_status = perf_eval.get('status', 'info')

                if perf_status == 'success':
                    st.success(f"▲ {perf_label} (Skor Akumulasi: {comp_score:.1f} / 100)", icon=":material/thumb_up:")
                elif perf_status == 'info':
                    st.info(f"▲ {perf_label} (Skor Akumulasi: {comp_score:.1f} / 100)", icon=":material/thumb_up:")
                elif perf_status == 'warning':
                    st.warning(f"▼ {perf_label} (Skor Akumulasi: {comp_score:.1f} / 100)", icon=":material/thumb_down:")
                else:
                    st.error(f"▼ {perf_label}", icon=":material/thumb_down:")

                actual_dates = data.index[-len(y_test):]

                # Plot Interaktif Evaluasi
                fig_eval = plot_interactive_evaluation(actual_dates, y_test_original, y_pred, f'Harga {asset_type}', curr_prefix=curr_prefix)
                if fig_eval is not None:
                    render_plotly_with_tools(fig_eval, key="fig_eval_chart")

                with st.expander(f"📈 Grafik Tren Riwayat Harga Candlestick, VPVR, Volume, ATR & Delta Volume (Data Pengujian)", expanded=False):
                    test_df = data.iloc[-len(y_test):]
                    fig_eval_comp = plot_comprehensive_market_indicators(test_df, f'Indikator Pasar Komprehensif Candlestick & VPVR (Data Pengujian {asset_type})', curr_prefix=curr_prefix, asset_type=asset_type)
                    if fig_eval_comp is not None:
                        render_plotly_with_tools(fig_eval_comp, key="fig_eval_comp_indicators")

                with st.popover("Tampilkan Data Pengujian & Hasil Prediksi Uji"):
                    eval_df = pd.DataFrame({
                        'Tanggal': [format_timestamp_for_plot(d) for d in actual_dates],
                        'Harga Aktual': [smart_format(v, prefix=curr_prefix) for v in y_test_original.flatten()],
                        'Prediksi Uji Model': [smart_format(v, prefix=curr_prefix) for v in y_pred.flatten()]
                    })
                    st.dataframe(eval_df, use_container_width=True)
                    st.download_button(
                        label="📥 Unduh Data Pengujian sebagai CSV",
                        data=eval_df.to_csv(index=False).encode('utf-8'),
                        file_name=f"evaluasi_pengujian_{stock}.csv",
                        mime="text/csv",
                        key="btn_dl_eval_data"
                    )

                with st.popover("Tampilkan Detail Metrik Tambahan"):
                    st.markdown(f"- **NRMSE (Normalized RMSE):** {perf_eval.get('nrmse', 0):.4f} ({perf_eval.get('nrmse_pct', 0):.2f}% dari rata-rata)")
                    mda_val = perf_eval.get('mda')
                    mda_str = f"{mda_val:.2f}%" if mda_val is not None else "-"
                    st.markdown(f"- **MDA (Mean Directional Accuracy):** {mda_str}")
                    st.markdown(f"- **Jumlah Sampel Uji:** {len(y_test)} langkah data")

                with st.popover("Menampilkan Grafik Loss dan Val Loss"):
                    final_loss = history.history['loss'][-1]
                    final_val_loss = history.history['val_loss'][-1]
                    st.metric("Loss akhir", smart_format(final_loss, default_decimals=4))
                    st.metric("Validation Loss akhir", smart_format(final_val_loss, default_decimals=4))
                    st.subheader("Riwayat Pelatihan")
                    st.line_chart(pd.DataFrame(history.history))

                st.success("Evaluasi Model selesai!")
        else:
            st.warning('Harus Melakukan Pelatihan Model Terlebih dahulu', icon=":material/exclamation:")

    with st.expander("7. Visualisasi Prediksi dan Perhitungan Metrik"):

        def forecast_future(model, last_sequence, scaler, n_steps):
            forecast = []
            curr_seq = np.array(last_sequence, dtype=np.float32).copy()
            seq_len = curr_seq.shape[0]
            
            seq_diffs = np.diff(curr_seq.flatten())
            std_step = float(np.std(seq_diffs)) if len(seq_diffs) > 0 else 0.02
            step_bound = max(0.015, min(0.08, std_step * 3.5))

            curr_arr = curr_seq.reshape(seq_len)
            prev_val = float(curr_arr[-1])
            curr_tensor = tf.constant(curr_arr.reshape(1, seq_len, 1), dtype=tf.float32)

            for step_idx in range(n_steps):
                pred_tensor = model(curr_tensor, training=False)
                raw_pred = float(pred_tensor.numpy()[0, 0])
                
                step_delta = raw_pred - prev_val
                clipped_delta = np.clip(step_delta, -step_bound, step_bound)
                
                if step_idx > 45:
                    damp_factor = 1.0 / (1.0 + (step_idx - 45) * 0.015)
                    clipped_delta *= damp_factor
                    
                final_step_val = float(np.clip(prev_val + clipped_delta, 0.005, 0.995))
                forecast.append(final_step_val)
                
                curr_arr = np.roll(curr_arr, -1)
                curr_arr[-1] = final_step_val
                curr_tensor = tf.constant(curr_arr.reshape(1, seq_len, 1), dtype=tf.float32)
                prev_val = final_step_val

            forecast = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))
            return forecast

        if btn_check == 1:

            start_time = time.time()
            progress_bar = st.progress(0)
            status_text = st.empty()
            time_estimate = st.empty()

            with st.spinner('Sedang Melakukan prediksi dan perhitungan metrik... Harap tunggu.'):

                for i, forecast_period in enumerate(selected_periods):

                    forecast_days = forecast_options_dict[forecast_period]
                    last_sequence = x_test[-1]
                    forecast = forecast_future(model, last_sequence, scaler, forecast_days)

                    progress = (i + 1) / len(selected_periods)
                    progress_bar.progress(progress)

                    elapsed_time = time.time() - start_time
                    estimated_total_time = elapsed_time / progress if progress > 0 else 0
                    remaining_time = estimated_total_time - elapsed_time

                    time_estimate.text(f"Estimasi waktu tersisa: {remaining_time:.2f} detik")
                    last_date = data.index[-1]
                    
                    if is_cmc:
                        if len(data) >= 2:
                            step_td = data.index[-1] - data.index[-2]
                            if step_td <= pd.Timedelta(0):
                                step_td = pd.Timedelta(minutes=1)
                        else:
                            step_td = pd.Timedelta(minutes=1)
                        date_range = [last_date + (j + 1) * step_td for j in range(forecast_days)]
                    else:
                        date_range = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)

                    st.subheader(f"Prediksi untuk {forecast_period}:")

                    start_idx = max(-len(data), -(forecast_days * 3))

                    fig_fc = plot_interactive_forecast(
                        hist_dates=data.index[start_idx:],
                        hist_prices=data['Close'].values[start_idx:],
                        actual_dates=actual_dates[start_idx:],
                        y_pred=y_pred[start_idx:],
                        date_range=date_range,
                        forecast=forecast,
                        y_label=f'Harga {asset_type}',
                        curr_prefix=curr_prefix
                    )
                    if fig_fc is not None:
                        render_plotly_with_tools(fig_fc, key=f"fig_forecast_{forecast_period}_{i}")

                    with st.expander(f"📈 Grafik Tren Riwayat Harga Candlestick, VPVR, Volume, ATR & Delta Volume ({forecast_period})", expanded=False):
                        fc_slice_df = data.iloc[start_idx:]
                        fig_fc_comp = plot_comprehensive_market_indicators(fc_slice_df, f'Indikator Pasar Komprehensif Candlestick & VPVR (Periode {forecast_period} {asset_type})', curr_prefix=curr_prefix, asset_type=asset_type)
                        if fig_fc_comp is not None:
                            render_plotly_with_tools(fig_fc_comp, key=f"fig_forecast_comp_{forecast_period}_{i}")

                    last_actual_price = safe_float(data['Close'].iloc[-1])
                    last_test_price = safe_float(y_pred[-1]) if (len(y_pred) > 0) else last_actual_price
                    last_forecast_price = safe_float(forecast[-1][0])
                    
                    pct_model_change = ((last_forecast_price - last_test_price) / last_test_price) * 100.0 if last_test_price > 0 else 0.0
                    last_proj_price = last_actual_price * (1.0 + (pct_model_change / 100.0))
                    pct_proj_change = ((last_proj_price - last_actual_price) / last_actual_price) * 100.0 if last_actual_price > 0 else 0.0
                    percent_change = pct_proj_change

                    if len(y_test) >= forecast_days:

                        st.subheader("Tabel dan Metrik Performa:")

                        table_df = pd.DataFrame({
                            'Tanggal': [format_timestamp_for_plot(d) for d in date_range],
                            'Harga Prediksi': [smart_format(v, prefix=curr_prefix) for v in forecast.flatten()]
                        })

                        perf_eval_sub = evaluate_model_performance(y_test_original[:forecast_days], y_pred[:forecast_days]) or {}

                        p_col1, p_col2, p_col3 = st.columns(3)
                        with p_col1:
                            render_colored_metric_card("Akurasi Horizon", f"{perf_eval_sub.get('accuracy', 0):.3f}%", perf_eval_sub.get('cat_acc', 'neutral'), f"Ketepatan {forecast_period}")
                        with p_col2:
                            r2_sub_disp = f"{perf_eval_sub.get('r2'):.3f}" if perf_eval_sub.get('r2') is not None else "-"
                            render_colored_metric_card("R2 Score", r2_sub_disp, perf_eval_sub.get('cat_r2', 'neutral'), "Koefisien determinasi")
                        with p_col3:
                            render_colored_metric_card("MAPE", f"{perf_eval_sub.get('mape', 0):.3f}", perf_eval_sub.get('cat_mape', 'neutral'), "Mean Abs Percentage Err")

                        p_col4, p_col5, p_col6 = st.columns(3)
                        with p_col4:
                            render_colored_metric_card("RMSE", smart_format(perf_eval_sub.get('rmse', 0), default_decimals=3), perf_eval_sub.get('cat_rmse', 'neutral'), "Root Mean Sq Err")
                        with p_col5:
                            render_colored_metric_card("MAE", smart_format(perf_eval_sub.get('mae', 0), default_decimals=3), perf_eval_sub.get('cat_mae', 'neutral'), "Mean Absolute Error")
                        with p_col6:
                            render_colored_metric_card("MSE", smart_format(perf_eval_sub.get('mse', 0), default_decimals=3), perf_eval_sub.get('cat_mse', 'neutral'), "Mean Squared Error")

                        with st.popover("Tampilkan Tabel Prediksi"):
                            st.dataframe(table_df, use_container_width=True)
                            st.download_button(
                                label="📥 Unduh Tabel Prediksi sebagai CSV",
                                data=table_df.to_csv(index=False).encode('utf-8'),
                                file_name=f"prediksi_{stock}_{forecast_period}.csv",
                                mime="text/csv",
                                key=f"btn_dl_pred_{forecast_period}_{i}"
                            )

                        st.subheader("Ringkasan Prediksi & Proyeksi Tren")

                        st.markdown("**1. Proyeksi Pasar Riil (Disesuaikan terhadap Harga Aktual):**")
                        r_col1, r_col2 = st.columns(2)
                        with r_col1:
                            st.metric("Harga Terakhir Aktual", smart_format(last_actual_price, prefix=curr_prefix))
                        with r_col2:
                            st.metric(
                                "Proyeksi Tren Aktual",
                                smart_format(last_proj_price, prefix=curr_prefix),
                                f"{pct_proj_change:+.2f}%"
                            )

                        st.markdown("**2. Estimasi Output Model Neural Network (Basis Data Uji):**")
                        r_col3, r_col4 = st.columns(2)
                        with r_col3:
                            st.metric("Harga Terakhir Pengujian (Uji)", smart_format(last_test_price, prefix=curr_prefix))
                        with r_col4:
                            st.metric(
                                "Prediksi Harga Model",
                                smart_format(last_forecast_price, prefix=curr_prefix),
                                f"{pct_model_change:+.2f}%"
                            )

                        comp_sub = perf_eval_sub.get('composite_score', 0)
                        sub_lbl = perf_eval_sub.get('label', 'Performa')
                        sub_stat = perf_eval_sub.get('status', 'info')

                        if sub_stat == 'success':
                            st.success(f"▲ {sub_lbl} (Skor Akumulasi: {comp_sub:.1f} / 100)", icon=":material/thumb_up:")
                        elif sub_stat == 'info':
                            st.info(f"▲ {sub_lbl} (Skor Akumulasi: {comp_sub:.1f} / 100)", icon=":material/thumb_up:")
                        elif sub_stat == 'warning':
                            st.warning(f"▼ {sub_lbl} (Skor Akumulasi: {comp_sub:.1f} / 100)", icon=":material/thumb_down:")
                        else:
                            st.error(f"▼ {sub_lbl}", icon=":material/thumb_down:")

                    else:
                        st.warning(f"Data tidak cukup untuk periode {forecast_period}, silahkan atur kembali rentang waktu pelatihan pada 'Pengumpulan data'.", icon=":material/exclamation:")

                    st.write("---")

            end_time = time.time()

            st.success(f"Prediksi dan perhitungan metrik selesai! Waktu komputasi total: {end_time - start_time:.2f} detik")

        else:
            st.warning('Harus Melakukan Pelatihan Model Terlebih dahulu', icon=":material/exclamation:")

    with st.expander("8. Interpretasi dan Pelaporan Hasil", True):

        if btn_check == 1:

            if 'last_actual_price' in locals() and 'last_forecast_price' in locals() and 'percent_change' in locals():

                st.subheader(f"Ringkasan Prediksi & Proyeksi **{forecast_period}** ke depan")
                
                st.markdown("**Proyeksi Pasar Riil:**")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Harga Terakhir Aktual", smart_format(last_actual_price, prefix=curr_prefix))
                with col2:
                    st.metric("Proyeksi Tren Aktual", smart_format(last_proj_price, prefix=curr_prefix), f"{pct_proj_change:+.2f}%")

                st.markdown("**Estimasi Output Model:**")
                col3, col4 = st.columns(2)
                with col3:
                    st.metric("Harga Terakhir Pengujian (Uji)", smart_format(last_test_price, prefix=curr_prefix))
                with col4:
                    st.metric("Prediksi Harga Model", smart_format(last_forecast_price, prefix=curr_prefix), f"{pct_model_change:+.2f}%")

                def interpret_forecast(percent_change):
                    if percent_change < -20:
                        return f"Tren harga {asset_type.lower()} diprediksi akan sangat turun 🔴."
                    elif percent_change < -5:
                        return f"Tren harga {asset_type.lower()} diprediksi akan turun 🟠."
                    elif percent_change < 5:
                        return f"Harga {asset_type.lower()} diprediksi akan stabil ⚫."
                    elif percent_change < 20:
                        return f"Tren harga {asset_type.lower()} diprediksi akan naik 🟡."
                    else:
                        return f"Tren harga {asset_type.lower()} diprediksi akan sangat naik 🟢."

                interpretation = interpret_forecast(percent_change)

                st.write(interpretation)

                if perf_eval['status'] == 'success':
                    st.success(f"{perf_eval['label']} (Skor Akumulasi: {perf_eval['composite_score']:.1f} / 100)", icon=":material/thumb_up:")
                elif perf_eval['status'] == 'info':
                    st.info(f"{perf_eval['label']} (Skor Akumulasi: {perf_eval['composite_score']:.1f} / 100)", icon=":material/thumb_up:")
                elif perf_eval['status'] == 'warning':
                    st.warning(f"{perf_eval['label']} (Skor Akumulasi: {perf_eval['composite_score']:.1f} / 100)", icon=":material/thumb_down:")
                else:
                    st.error(f"{perf_eval['label']}", icon=":material/thumb_down:")

                def analyze_market_trends(data, forecast):
                    recent_trend = "bullish" if data['Close'].pct_change().mean().item() > 0 else "bearish"
                    forecast_trend = "naik" if forecast[-1] > forecast[0] else "turun"
                    return f"Tren pasar terkini cenderung {recent_trend}. Berdasarkan prediksi, harga {asset_type.lower()} diperkirakan akan {forecast_trend} dalam periode mendatang."

                def generate_recommendation(percent_change, perf_res):
                    acc = perf_res['accuracy']
                    stat = perf_res['status']
                    if stat == 'success' and acc >= 85:
                        if percent_change < -5:
                            return f"Prediksi menunjukkan tren penurunan pada {asset_type.lower()} dengan tingkat akurasi model yang tinggi. Waspadai risiko penurunan harga dan pertimbangkan strategi manajemen risiko."
                        elif percent_change > 5:
                            return f"Prediksi menunjukkan tren kenaikan pada {asset_type.lower()} dengan tingkat akurasi model yang tinggi. Peluang momentum positif terlihat baik."
                        else:
                            return f"Prediksi menunjukkan harga {asset_type.lower()} relatif stabil dengan akurasi model yang tinggi."
                    elif stat in ['info', 'warning']:
                        return f"Model menunjukkan akurasi moderat ({acc:.1f}%). Gunakan hasil prediksi sebagai indikator pelengkap bersama analisis teknikal lainnya."
                    else:
                        return f"Tingkat akurasi model saat ini rendah ({acc:.1f}%). Disarankan menambah jumlah data pelatihan atau menyesuaikan hyperparameter (epoch/batch) sebelum mengambil keputusan investasi."
            else:
                return
                    
            st.subheader("Insight Pasar")
            market_trends = analyze_market_trends(data, forecast)
            st.write(market_trends)
            
            with st.expander("💡 Rekomendasi & Catatan Penting", expanded=False):
                st.markdown("**Rekomendasi:**")
                recommendation = generate_recommendation(percent_change, perf_eval)
                st.write(recommendation)
                
                st.markdown("**Catatan Penting:**")
                st.warning(f"""
- Prediksi ini didasarkan pada data historis dan model statistik CNN-GRU.
- Faktor eksternal seperti kondisi makroekonomi, kebijakan moneter, dan sentimen pasar global dapat mempengaruhi harga {asset_type.lower()} secara signifikan.
- Selalu lakukan analisis fundamental & teknikal mandiri sebelum membuat keputusan investasi.
                """, icon=":material/edit_note:")
        else:
            st.warning('Harus Melakukan Pelatihan Model Terlebih dahulu', icon=":material/exclamation:")

if __name__ == "__main__":
    
    manual_select_type = None
    if st.session_state.get('redirect_to_input_custom', False):
        st.session_state.menu_type_index = 1
        manual_select_type = 1
        st.session_state.redirect_to_input_custom = False
        
    if 'menu_type_index' not in st.session_state:
        st.session_state.menu_type_index = 0

    with st.sidebar:
        st.write("<h1 style='text-align: left'><b>DASHBOARD PREDIKSI SAHAM & CRYPTO DENGAN CNN-GRU</b></h1>", unsafe_allow_html=True)
        
        st.write("\n")
        
        st.markdown('**PILIH MENU**')
        
        menu_type = option_menu(
            menu_title=None,
            options=["Informasi Umum", "Prediksi Saham"],
            icons=["info-circle", "graph-up"],
            default_index=st.session_state.menu_type_index,
            manual_select=manual_select_type,
            orientation="horizontal"
        )
        
        menu_type_options = ["Informasi Umum", "Prediksi Saham"]
        if menu_type in menu_type_options:
            st.session_state.menu_type_index = menu_type_options.index(menu_type)
        
        if menu_type == "Informasi Umum":
            if 'selected_index_info' not in st.session_state:
                st.session_state.selected_index_info = 0
                
            selected = option_menu(
                menu_title=None,
                options=["Gambaran Umum", "Glosarium", "Metodologi"],
                icons=["house", "book", "pen"],
                default_index=st.session_state.selected_index_info,
                orientation="vertikal"
            )
            
            info_options = ["Gambaran Umum", "Glosarium", "Metodologi"]
            if selected in info_options:
                st.session_state.selected_index_info = info_options.index(selected)
            
        elif menu_type == "Prediksi Saham":
            manual_select_pred = None
            if st.session_state.get('redirect_to_input_custom_pred', False):
                st.session_state.selected_index_pred = 0
                manual_select_pred = 0
                st.session_state.redirect_to_input_custom_pred = False
                
            if 'selected_index_pred' not in st.session_state:
                st.session_state.selected_index_pred = 0
                
            selected = option_menu(
                menu_title=None,
                options=[
                    "Input Saham Custom",
                    "Input Crypto (CoinMarketCap)",
                    "PT Bank Mandiri Tbk (Bank Mandiri)",
                    "PT Bank Rakyat Indonesia Tbk (BRI)",
                    "PT Bank Central Asia Tbk (BCA)",
                    "PT Bank Negara Indonesia Tbk (BNI)",
                    "PT Bank Syariah Indonesia Tbk (BSI)"
                ],
                icons=["search", "currency-bitcoin", "bank", "bank", "bank", "bank", "bank"],
                default_index=st.session_state.selected_index_pred,
                manual_select=manual_select_pred,
                orientation="vertikal"
            )
            
            pred_options = [
                "Input Saham Custom",
                "Input Crypto (CoinMarketCap)",
                "PT Bank Mandiri Tbk (Bank Mandiri)",
                "PT Bank Rakyat Indonesia Tbk (BRI)",
                "PT Bank Central Asia Tbk (BCA)",
                "PT Bank Negara Indonesia Tbk (BNI)",
                "PT Bank Syariah Indonesia Tbk (BSI)"
            ]
            if selected in pred_options:
                st.session_state.selected_index_pred = pred_options.index(selected)
            
        st.markdown('**Manual**')
        st.markdown('- **1. Pilih Tab Prediksi Saham:** Untuk Melakukan Forecasting')
        st.markdown('- **2. Pilih Sumber Data & Simbol Aset:** yFinance (Harian) atau CoinMarketCap (Intraday Jam/Menit/Detik)')
        st.markdown('- **3. Scroll ke bawah halaman:** Untuk Memilih Periode Forecasting')
        st.markdown('- **4. Tekan Tombol Latih Model:** Untuk Melakukan Pelatihan Model Forecasting')
        st.markdown('- **5. Lihat Interpretasi dan Pelaporan Hasil:** Menampilkan Kesimpulan Prediksi')
        
        st.write("\n")
        
        with st.expander("Lainnya"):
            st.markdown('**Manual Tambahan**')
            st.markdown('- **Pilih Tab Informasi Umum:** Untuk Mengetahui Informasi Mengenai Aplikasi ini')
            st.markdown('- **Buka Toggle List Persiapan Lingkungan:** Untuk Melihat Library yang digunakan pada Aplikasi ini')
            st.markdown('- **Buka Toggle List Pengumpulan Data:** Untuk Memilih Jumlah Pelatihan Data')
            st.markdown('- **Buka Toggle List Pra-pemrsesan Data:** Untuk Memilih Jumlah Data Pelatihan dan Data Pengujian')
            st.markdown('- **Buka Toggle List Perancangan CNN-GRU:** Untuk Melihat Model Mesin Prediksi yang digunakan')
            st.markdown('- **Buka Toggle List Pelatihan Model:** Untuk Menguubah Hiperparameter, Memilih Beberapa Periode Forecasting dan Melukan Pelatihan Model')
            st.markdown('- **Buka Toggle List Evaluasi Model:** Untuk Melihat Kemampuan Model')
            st.markdown('- **Buka Toggle List Visualisasi Prediksi dan Perhitungan Metrik:** Untuk Melihat Hasil Prediksi Berupa Grafik Dan Perhitungan Metrik')
            st.markdown('- **Buka Toggle List Interpretasi dan Pelaporan Hasil:** Untuk Melihat Interpretasi dan Pelaporan Akhir Periode Forecast yang terakhir')

if menu_type == "Informasi Umum":
    
    if selected == "Gambaran Umum":
        with open('./TEXT/gambaran_umum.md', 'r', encoding='utf-8') as file:
            html_content = file.read()
        
        target_marker = "Portofolio ini dibuat oleh Ilham Rizkyansyah &middot; Universitas Gunadarma Informatika"
        if target_marker in html_content:
            parts = html_content.split(target_marker)
            subparts = parts[1].split("</p>", 1)
            
            st.markdown(parts[0] + target_marker + subparts[0] + "</p>", unsafe_allow_html=True)
            
            if st.button("👉 Klik di sini untuk melakukan forecasting langsung", type="primary", use_container_width=True):
                st.session_state.redirect_to_input_custom = True
                st.session_state.redirect_to_input_custom_pred = True
                st.rerun()
            
            st.markdown(subparts[1], unsafe_allow_html=True)
        else:
            st.markdown(html_content, unsafe_allow_html=True)
        
    elif selected == "Glosarium":
        with open('./TEXT/glosarium.md', 'r', encoding='utf-8') as file:
            html_content = file.read()
        
        st.markdown(html_content, unsafe_allow_html=True)
        
    elif selected == "Metodologi":
        with open('./TEXT/metodologi.md', 'r', encoding='utf-8') as file:
            html_content = file.read()
    
        st.markdown(html_content, unsafe_allow_html=True)  
        
if menu_type == "Prediksi Saham":
    
    if selected == "Input Saham Custom":
        if 'custom_stock_input' not in st.session_state:
            st.session_state.custom_stock_input = "BTC-USD"

        st.markdown("<h1 style='text-align: left; color: #4A4A4A;'>Input Saham / Crypto Custom (yFinance)</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: justify; color: black;'>Masukkan kode saham atau crypto yang ingin Anda prediksi via yFinance (contoh: AAPL, GOOGL, BMRI.JK, BTC-USD, dll.)</p>", unsafe_allow_html=True)
        st.info("💡 **Tips:** Silakan scroll ke bawah halaman untuk melakukan forecasting/prediksi setelah memilih atau memasukkan kode saham.")
        
        # Popular Indonesia Stock Buttons
        st.write("**Pilihan Saham Indonesia Populer**")
        ind_cols = st.columns(4)
        with ind_cols[0]:
            if st.button("BBCA.JK", key="btn_bbca"):
                st.session_state.custom_stock_input = "BBCA.JK"
                st.rerun()
        with ind_cols[1]:
            if st.button("BMRI.JK", key="btn_bmri"):
                st.session_state.custom_stock_input = "BMRI.JK"
                st.rerun()
        with ind_cols[2]:
            if st.button("BBNI.JK", key="btn_bbni"):
                st.session_state.custom_stock_input = "BBNI.JK"
                st.rerun()
        with ind_cols[3]:
            if st.button("BRIS.JK", key="btn_bris"):
                st.session_state.custom_stock_input = "BRIS.JK"
                st.rerun()
        
        # Popular Global Stock Buttons
        st.write("**Pilihan Saham Global Populer**")
        glob_cols1 = st.columns(4)
        with glob_cols1[0]:
            if st.button("AAPL", key="btn_aapl"):
                st.session_state.custom_stock_input = "AAPL"
                st.rerun()
        with glob_cols1[1]:
            if st.button("GOOGL", key="btn_googl"):
                st.session_state.custom_stock_input = "GOOGL"
                st.rerun()
        with glob_cols1[2]:
            if st.button("MSFT", key="btn_msft"):
                st.session_state.custom_stock_input = "MSFT"
                st.rerun()
        with glob_cols1[3]:
            if st.button("TSLA", key="btn_tsla"):
                st.session_state.custom_stock_input = "TSLA"
                st.rerun()
        
        glob_cols2 = st.columns(4)
        with glob_cols2[0]:
            if st.button("AMZN", key="btn_amzn"):
                st.session_state.custom_stock_input = "AMZN"
                st.rerun()
        with glob_cols2[1]:
            if st.button("NVDA", key="btn_nvda"):
                st.session_state.custom_stock_input = "NVDA"
                st.rerun()
        with glob_cols2[2]:
            if st.button("META", key="btn_meta"):
                st.session_state.custom_stock_input = "META"
                st.rerun()
        with glob_cols2[3]:
            if st.button("NFLX", key="btn_nflx"):
                st.session_state.custom_stock_input = "NFLX"
                st.rerun()
        
        # Popular Crypto Buttons
        st.write("**Pilihan Crypto Populer (Global)**")
        crypto_cols1 = st.columns(4)
        with crypto_cols1[0]:
            if st.button("BTC-USD", key="btn_btc"):
                st.session_state.custom_stock_input = "BTC-USD"
                st.rerun()
        with crypto_cols1[1]:
            if st.button("ETH-USD", key="btn_eth"):
                st.session_state.custom_stock_input = "ETH-USD"
                st.rerun()
        with crypto_cols1[2]:
            if st.button("SOL-USD", key="btn_sol"):
                st.session_state.custom_stock_input = "SOL-USD"
                st.rerun()
        with crypto_cols1[3]:
            if st.button("BNB-USD", key="btn_bnb"):
                st.session_state.custom_stock_input = "BNB-USD"
                st.rerun()
        
        crypto_cols2 = st.columns(4)
        with crypto_cols2[0]:
            if st.button("XRP-USD", key="btn_xrp"):
                st.session_state.custom_stock_input = "XRP-USD"
                st.rerun()
        with crypto_cols2[1]:
            if st.button("DOGE-USD", key="btn_doge"):
                st.session_state.custom_stock_input = "DOGE-USD"
                st.rerun()
        with crypto_cols2[2]:
            if st.button("ADA-USD", key="btn_ada"):
                st.session_state.custom_stock_input = "ADA-USD"
                st.rerun()
        with crypto_cols2[3]:
            if st.button("DOT-USD", key="btn_dot"):
                st.session_state.custom_stock_input = "DOT-USD"
                st.rerun()
        
        with st.expander("💡 Ketentuan & Tips Penulisan Ticker yFinance", expanded=False):
            st.markdown("""
            **Ketentuan Penulisan Ticker yFinance:**
            1. **Saham Indonesia (IHSG)**: Tambahkan akhiran **`.JK`** (misalnya: `BBCA.JK`, `BMRI.JK`, `BBNI.JK`, `BRIS.JK`).
            2. **Saham Global (AS/S&P 500)**: Kode ticker standar bursa AS (misalnya: `AAPL`, `GOOGL`, `MSFT`, `TSLA`).
            3. **Cryptocurrency**: Gunakan kode koin diikuti akhiran **`-USD`** (misalnya: `BTC-USD`, `ETH-USD`, `SOL-USD`).
            """)
        
        st.write("")
        st.markdown("**Masukkan Kode Saham/Crypto**")
        col_input, col_btn = st.columns([5, 1])
        with col_input:
            custom_stock = st.text_input(
                "Masukkan Kode Saham", 
                value=st.session_state.custom_stock_input, 
                placeholder="Contoh: BMRI.JK", 
                label_visibility="collapsed"
            )
        with col_btn:
            st.button("Enter 🔍", type="primary", use_container_width=True)
            
        if custom_stock != st.session_state.custom_stock_input:
            st.session_state.custom_stock_input = custom_stock
        
        if custom_stock:
            st.cache_data.clear()
            main(custom_stock, data_source="yfinance")
        else:
            st.warning("Silakan masukkan kode saham terlebih dahulu")

    elif selected == "Input Crypto (CoinMarketCap)":
        if 'cmc_crypto_input' not in st.session_state:
            st.session_state.cmc_crypto_input = "BTC"

        st.markdown("<h1 style='text-align: left; color: #F7931A;'>Prediksi Crypto melalui CoinMarketCap</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: justify; color: black;'>Masukkan kode aset crypto dan API Key CoinMarketCap untuk analisis data frekuensi tinggi (Menit/Jam/Detik).</p>", unsafe_allow_html=True)
        
        default_cmc_key = get_cmc_api_key_from_env_or_secrets()
        cmc_key_input = st.text_input("🔑 Masukkan CoinMarketCap API Key", value=default_cmc_key, type="password", placeholder="Masukkan API Key CoinMarketCap Anda (Opsional jika sudah ada di secrets)...")
        if default_cmc_key:
            st.caption("✅ API Key terdeteksi dari Secrets/Environment.")
        
        st.write("**Pilihan Kripto Populer (CoinMarketCap)**")
        cmc_cols1 = st.columns(4)
        with cmc_cols1[0]:
            if st.button("BTC (Bitcoin)", key="cmc_btc"):
                st.session_state.cmc_crypto_input = "BTC"
                st.rerun()
        with cmc_cols1[1]:
            if st.button("ETH (Ethereum)", key="cmc_eth"):
                st.session_state.cmc_crypto_input = "ETH"
                st.rerun()
        with cmc_cols1[2]:
            if st.button("SOL (Solana)", key="cmc_sol"):
                st.session_state.cmc_crypto_input = "SOL"
                st.rerun()
        with cmc_cols1[3]:
            if st.button("BNB (Binance)", key="cmc_bnb"):
                st.session_state.cmc_crypto_input = "BNB"
                st.rerun()
                
        cmc_cols2 = st.columns(4)
        with cmc_cols2[0]:
            if st.button("XRP (Ripple)", key="cmc_xrp"):
                st.session_state.cmc_crypto_input = "XRP"
                st.rerun()
        with cmc_cols2[1]:
            if st.button("DOGE (Dogecoin)", key="cmc_doge"):
                st.session_state.cmc_crypto_input = "DOGE"
                st.rerun()
        with cmc_cols2[2]:
            if st.button("ADA (Cardano)", key="cmc_ada"):
                st.session_state.cmc_crypto_input = "ADA"
                st.rerun()
        with cmc_cols2[3]:
            if st.button("DOT (Polkadot)", key="cmc_dot"):
                st.session_state.cmc_crypto_input = "DOT"
                st.rerun()
                
        st.write("")
        st.markdown("**Masukkan Simbol Kripto**")
        col_c_in, col_c_btn = st.columns([5, 1])
        with col_c_in:
            cmc_symbol = st.text_input(
                "Masukkan Simbol Kripto",
                value=st.session_state.cmc_crypto_input,
                placeholder="Contoh: BTC, ETH, DOGE, SOL",
                label_visibility="collapsed"
            )
        with col_c_btn:
            st.button("Enter 🚀", type="primary", use_container_width=True, key="btn_cmc_enter")
            
        if cmc_symbol != st.session_state.cmc_crypto_input:
            st.session_state.cmc_crypto_input = cmc_symbol
            
        if cmc_symbol:
            st.cache_data.clear()
            main(cmc_symbol, data_source="coinmarketcap", api_key=cmc_key_input)
        else:
            st.warning("Silakan masukkan simbol crypto terlebih dahulu")
    
    elif selected == "PT Bank Mandiri Tbk (Bank Mandiri)":
        image = Image.open('./LOGO/BMRI.png')
        st.image(image, caption=None, width=500, clamp=False, channels="RGB", output_format="auto")
        
        st.markdown("<h1 style='text-align: left; color: #003A70;'>PT Bank Mandiri Tbk</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: justify; color: black;'>PT Bank Mandiri (Persero) Tbk adalah salah satu bank BUMN terbesar di Indonesia yang didirikan pada 2 Oktober 1998 sebagai hasil merger 4 bank pemerintah. Bank Mandiri terdaftar di Bursa Efek Indonesia dengan kode saham BMRI. Pemegang saham utamanya adalah Pemerintah Indonesia. Bank ini berfokus pada layanan korporasi, komersial, mikro & ritel, dan tresuri. Bank Mandiri memiliki jaringan cabang dan ATM yang luas di Indonesia serta terus mengembangkan ekosistem digital melalui Livin' by Mandiri.</p>", unsafe_allow_html=True)
        st.write('Informasi singkat:')
        st.markdown('- **Tanggal Didirikan:** 2 Oktober 1998')
        st.markdown('- **Kode Saham Bursa Efek Indonesia:** BMRI')
        st.markdown('- **Pemegang Saham Utama:** Pemerintah Indonesia (66,56%)')
        st.markdown('- **Fokus Pada:** Layanan korporasi, komersial, mikro & ritel, dan tresuri')
        st.markdown('- **Mengembangkan:** Ekosistem digital melalui Livin by Mandiri')
        
        st.cache_data.clear()
        main("BMRI.JK", data_source="yfinance")

    elif selected == "PT Bank Rakyat Indonesia Tbk (BRI)":
        image = Image.open('./LOGO/BBRI.png')
        st.image(image, caption=None, width=385, clamp=False, channels="RGB", output_format="auto")
        
        st.markdown("<h1 style='text-align: left; color: #00529C;'>PT Bank Rakyat Indonesia Tbk</h1>", unsafe_allow_html=True) 
        st.markdown("<p style='text-align: justify; color: black;'>PT Bank Rakyat Indonesia (Persero) Tbk (BRI) adalah bank BUMN terbesar di Indonesia yang didirikan pada 16 Desember 1895. BRI terdaftar di Bursa Efek Indonesia dengan kode saham BBRI. Pemegang saham utamanya adalah Pemerintah Indonesia. BRI berfokus utama pada pembiayaan UMKM dan sektor pertanian. Bank ini memiliki jaringan unit kerja terluas hingga ke pelosok desa dan terus mengembangkan layanan perbankan digital seperti BRImo.</p>", unsafe_allow_html=True)
        st.write('Informasi singkat:')
        st.markdown('- **Tanggal Didirikan:** 16 Desember 1895')
        st.markdown('- **Kode Saham Bursa Efek Indonesia:** BBRI')
        st.markdown('- **Pemegang Saham Utama:** Pemerintah Indonesia (53,20%)')
        st.markdown('- **Fokus utama Pada:** Pembiayaan UMKM dan sektor pertanian')
        st.markdown('- **Mengembangkan:** Layanan perbankan digital seperti BRImo')
        
        st.cache_data.clear()
        main("BBRI.JK", data_source="yfinance")

    elif selected == "PT Bank Central Asia Tbk (BCA)":
        image = Image.open('./LOGO/BBCA.png')
        st.image(image, caption=None, width=465, clamp=False, channels="RGB", output_format="auto")
        
        st.markdown("<h1 style='text-align: left; color: #0060AF;'>PT Bank Central Asia Tbk</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: justify; color: black;'>PT Bank Central Asia Tbk adalah bank swasta terbesar di Indonesia yang didirikan pada 21 Februari 1957. BCA terdaftar di Bursa Efek Indonesia dengan kode saham BBCA. Pemegang saham utamanya adalah PT Dwimuria Investama Andalan. BCA berfokus pada layanan perbankan ritel, UKM, dan korporasi. Bank ini memiliki jaringan cabang dan ATM yang luas di seluruh Indonesia serta dikenal dengan layanan perbankan digitalnya seperti m-BCA dan KlikBCA.</p>", unsafe_allow_html=True)
        st.write('Informasi singkat:')
        st.markdown('- **Tanggal Didirikan:** 21 Februari 1957')
        st.markdown('- **Kode Saham Bursa Efek Indonesia:** BBCA')
        st.markdown('- **Pemegang Saham Utama:** PT Dwimuria Investama Andalan (54,94%)')
        st.markdown('- **Fokus Pada:** Layanan perbankan ritel, UKM, dan korporasi')
        st.markdown('- **Dikenal dengan:** Layanan perbankan digital seperti m-BCA dan KlikBCA')
        
        st.cache_data.clear()
        main("BBCA.JK", data_source="yfinance")  

    elif selected == "PT Bank Negara Indonesia Tbk (BNI)":
        image = Image.open('./LOGO/BBNI.png')
        st.image(image, caption=None, width=500, clamp=False, channels="RGB", output_format="auto")
        
        st.markdown("<h1 style='text-align: left; color: #006885;'>PT Bank Negara Indonesia Tbk</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: justify; color: black;'>PT Bank Negara Indonesia (Persero) Tbk adalah salah satu bank BUMN terbesar di Indonesia yang didirikan pada 5 Juli 1946. BNI terdaftar di Bursa Efek Indonesia dengan kode saham BBNI. Pemegang saham utamanya adalah Pemerintah Indonesia. BNI berfokus pada layanan korporasi, ritel, dan internasional. Bank ini memiliki jaringan cabang di dalam dan luar negeri serta terus mengembangkan layanan digital seperti BNI Mobile Banking.</p>", unsafe_allow_html=True)
        st.write('Informasi singkat:')
        st.markdown('- **Tanggal Didirikan:** 5 Juli 1946')
        st.markdown('- **Kode Saham Bursa Efek Indonesia:** BBNI')
        st.markdown('- **Pemegang Saham Utama:** Pemerintah Indonesia (60%)')
        st.markdown('- **Fokus Pada:** Layanan korporasi, ritel, dan internasional')
        st.markdown('- **Mengembangkan:** Layanan digital seperti BNI Mobile Banking')
        
        st.cache_data.clear()
        main("BBNI.JK", data_source="yfinance")

    elif selected == "PT Bank Syariah Indonesia Tbk (BSI)":
        image = Image.open('./LOGO/BRIS.png')
        st.image(image, caption=None, width=520, clamp=False, channels="RGB", output_format="auto")
        
        st.markdown("<h1 style='text-align: left; color: #00A39D;'>PT Bank Syariah Indonesia Tbk</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: justify; color: black;'>PT Bank Syariah Indonesia Tbk adalah bank syariah terbesar di Indonesia yang didirikan pada 1 Februari 2021 sebagai hasil merger 3 bank syariah BUMN. BSI terdaftar di Bursa Efek Indonesia dengan kode saham BRIS. Pemegang saham utamanya adalah PT Bank Mandiri (Persero) Tbk, PT Bank Negara Indonesia (Persero) Tbk, dan PT Bank Rakyat Indonesia (Persero) Tbk. BSI berfokus pada layanan perbankan syariah ritel dan korporasi serta terus mengembangkan ekosistem keuangan syariah digital.</p>", unsafe_allow_html=True)
        st.write('Informasi singkat:')
        st.markdown('- **Tanggal Didirikan:** 1 Februari 2021 (hasil merger 3 bank syariah BUMN)')
        st.markdown('- **Kode Saham Bursa Efek Indonesia:** BRIS')
        st.markdown('- **Pemegang Saham Utama:** PT Bank Mandiri (Persero) Tbk (51,47%), PT Bank Negara Indonesia (Persero) Tbk (23,24%), PT Bank Rakyat Indonesia (Persero) Tbk (15,38%)')
        st.markdown('- **Fokus Pada:** Layanan perbankan syariah ritel dan korporasi')
        st.markdown('- **Mengembangkan:** Ekosistem keuangan syariah digital')
        
        st.cache_data.clear()
        main("BRIS.JK", data_source="yfinance")
