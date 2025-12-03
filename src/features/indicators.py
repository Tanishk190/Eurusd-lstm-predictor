"""
Technical Indicators Module for EUR/USD Prediction
Provides various technical analysis indicators for feature engineering
"""

import pandas as pd
import numpy as np


def calculate_rsi(series, window=14):
    """
    Calculate Relative Strength Index (RSI)
    
    Args:
        series: Price series (typically Close prices)
        window: Period for RSI calculation (default: 14)
    
    Returns:
        RSI values as pandas Series
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(series, fast=12, slow=26, signal=9):
    """
    Calculate MACD (Moving Average Convergence Divergence)
    
    Args:
        series: Price series (typically Close prices)
        fast: Fast EMA period (default: 12)
        slow: Slow EMA period (default: 26)
        signal: Signal line period (default: 9)
    
    Returns:
        DataFrame with MACD, Signal, and Histogram
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    
    return pd.DataFrame({
        'MACD': macd,
        'MACD_Signal': signal_line,
        'MACD_Hist': histogram
    })


def calculate_bollinger_bands(series, window=20, num_std=2):
    """
    Calculate Bollinger Bands
    
    Args:
        series: Price series (typically Close prices)
        window: Period for moving average (default: 20)
        num_std: Number of standard deviations (default: 2)
    
    Returns:
        DataFrame with Upper, Middle, and Lower bands plus Bandwidth
    """
    middle = series.rolling(window=window).mean()
    std = series.rolling(window=window).std()
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    bandwidth = (upper - lower) / middle
    
    return pd.DataFrame({
        'BB_Upper': upper,
        'BB_Middle': middle,
        'BB_Lower': lower,
        'BB_Bandwidth': bandwidth
    })


def calculate_ema(series, span):
    """
    Calculate Exponential Moving Average
    
    Args:
        series: Price series
        span: Period for EMA
    
    Returns:
        EMA values as pandas Series
    """
    return series.ewm(span=span, adjust=False).mean()


def calculate_sma(series, window):
    """
    Calculate Simple Moving Average
    
    Args:
        series: Price series
        window: Period for SMA
    
    Returns:
        SMA values as pandas Series
    """
    return series.rolling(window=window).mean()


def calculate_atr(high, low, close, window=14):
    """
    Calculate Average True Range (ATR)
    
    Args:
        high: High price series
        low: Low price series
        close: Close price series
        window: Period for ATR (default: 14)
    
    Returns:
        ATR values as pandas Series
    """
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = true_range.rolling(window=window).mean()
    
    return atr


def calculate_momentum(series, window=10):
    """
    Calculate Momentum (Rate of Change)
    
    Args:
        series: Price series
        window: Period for momentum calculation (default: 10)
    
    Returns:
        Momentum values as pandas Series
    """
    return series.diff(window)


def add_all_indicators(df):
    """
    Add all technical indicators to a dataframe
    
    Args:
        df: DataFrame with OHLC data (Open, High, Low, Close)
    
    Returns:
        DataFrame with all technical indicators added
    """
    data = pd.DataFrame(index=df.index)
    
    # Price data
    if 'Close' in df.columns:
        data['Close'] = df['Close']
    elif 'Adj Close' in df.columns:
        data['Close'] = df['Adj Close']
    else:
        raise ValueError("No Close price column found")
    
    # Moving Averages
    data['SMA_5'] = calculate_sma(data['Close'], 5)
    data['SMA_10'] = calculate_sma(data['Close'], 10)
    data['SMA_20'] = calculate_sma(data['Close'], 20)
    data['SMA_50'] = calculate_sma(data['Close'], 50)
    
    data['EMA_12'] = calculate_ema(data['Close'], 12)
    data['EMA_26'] = calculate_ema(data['Close'], 26)
    
    # RSI
    data['RSI'] = calculate_rsi(data['Close'])
    
    # MACD
    macd_df = calculate_macd(data['Close'])
    data['MACD'] = macd_df['MACD']
    data['MACD_Signal'] = macd_df['MACD_Signal']
    data['MACD_Hist'] = macd_df['MACD_Hist']
    
    # Bollinger Bands
    bb_df = calculate_bollinger_bands(data['Close'])
    data['BB_Upper'] = bb_df['BB_Upper']
    data['BB_Middle'] = bb_df['BB_Middle']
    data['BB_Lower'] = bb_df['BB_Lower']
    data['BB_Bandwidth'] = bb_df['BB_Bandwidth']
    
    # ATR (if High/Low available)
    if 'High' in df.columns and 'Low' in df.columns:
        data['ATR'] = calculate_atr(df['High'], df['Low'], data['Close'])
    
    # Momentum
    data['Momentum'] = calculate_momentum(data['Close'])
    
    # Price-based features
    data['Daily_Return'] = data['Close'].pct_change()
    data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))
    
    return data
