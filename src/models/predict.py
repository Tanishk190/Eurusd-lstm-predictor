"""
LSTM Prediction Script for EUR/USD
Uses trained LSTM model with technical indicators
"""

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import joblib
import os
from tensorflow import keras
from src.features.indicators import add_all_indicators


def predict_next_day(return_data=False):
    """
    Predict the next day's EUR/USD closing price using LSTM model
    
    Args:
        return_data: If True, returns prediction data as dict for API
    
    Returns:
        Dictionary with prediction data if return_data=True, None otherwise
    """
    print("=" * 60)
    print("EUR/USD Next-Day Price Prediction (LSTM)")
    print("=" * 60)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    models_dir = os.path.join(project_root, "models")
    
    # Load the trained model
    print("\n[1/5] Loading LSTM model...")
    model_file = os.path.join(models_dir, "eurusd_lstm_model.h5")
    
    try:
        model = keras.models.load_model(model_file)
        print(f"   [OK] Model loaded from {model_file}")
    except Exception as e:
        print(f"   [ERROR] Error loading model: {e}")
        print("   Please train the model first by running: python train_model.py")
        return None
    
    # Load the scaler
    print("\n[2/5] Loading scaler and feature configuration...")
    scaler_file = os.path.join(models_dir, "scaler.pkl")
    feature_file = os.path.join(models_dir, "feature_columns.pkl")
    
    try:
        scaler = joblib.load(scaler_file)
        feature_columns = joblib.load(feature_file)
        print(f"   [OK] Loaded {len(feature_columns)} features")
    except Exception as e:
        print(f"   [ERROR] Error loading files: {e}")
        return None
    
    # Fetch recent data
    print("\n[3/5] Fetching latest EUR/USD data...")
    df = yf.download("EURUSD=X", period="6mo", progress=False)
    
    # Handle MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.xs('EURUSD=X', level=1, axis=1)
        except:
            pass
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    print(f"   [OK] Fetched {len(df)} days of data")
    
    # Add technical indicators
    print("\n[4/5] Calculating technical indicators...")
    data = add_all_indicators(df)
    data = data.dropna()
    
    # Filter to use the same features as training
    dataset = data[feature_columns].values
    
    # Check if dataset is empty
    if dataset.shape[0] == 0:
        raise ValueError(f"Found array with 0 sample(s) (shape={dataset.shape}) while a minimum of 1 is required by MinMaxScaler.")
    
    # Scale the data
    scaled_data = scaler.transform(dataset)
    
    # We need 60 days of history to make a prediction
    seq_length = 60
    
    if len(scaled_data) < seq_length:
        print(f"   [ERROR] Not enough data. Need {seq_length} days, got {len(scaled_data)}")
        return None
    
    # Get the last 60 days
    X_input = scaled_data[-seq_length:]
    X_input = X_input.reshape(1, seq_length, len(feature_columns))
    
    # Make prediction
    print("\n[5/5] Generating prediction...")
    prediction_scaled = model.predict(X_input, verbose=0)
    
    # Inverse transform the prediction
    prediction_full = np.zeros((1, len(feature_columns)))
    prediction_full[0, 0] = prediction_scaled[0, 0]
    predicted_price = scaler.inverse_transform(prediction_full)[0, 0]
    
    # Get current information
    last_date = data.index[-1]
    current_close = data['Close'].iloc[-1]
    
    # Calculate next trading day
    next_date = last_date + pd.Timedelta(days=1)
    if next_date.weekday() >= 5:  # Saturday or Sunday
        next_date += pd.Timedelta(days=(7 - next_date.weekday()))
    
    change = predicted_price - current_close
    change_percent = (change / current_close) * 100
    
    # Display results
    print("\n" + "=" * 60)
    print("Prediction Results")
    print("=" * 60)
    print(f"Current Date:      {last_date.strftime('%Y-%m-%d')}")
    print(f"Current Close:     {current_close:.5f}")
    print(f"Predicted Date:    {next_date.strftime('%Y-%m-%d')}")
    print(f"Predicted Price:   {predicted_price:.5f}")
    print(f"Expected Change:   {change:+.5f} ({change_percent:+.2f}%)")
    print("=" * 60)
    
    if return_data:
        # Return data for API
        recent_data = data.iloc[-30:]
        history_dates = [d.strftime('%Y-%m-%d') for d in recent_data.index]
        history_prices = recent_data['Close'].tolist()
        
        return {
            "current_date": last_date.strftime('%Y-%m-%d'),
            "current_close": float(current_close),
            "predicted_date": next_date.strftime('%Y-%m-%d'),
            "predicted_price": float(predicted_price),
            "change": float(change),
            "change_percent": float(change_percent),
            "history_dates": history_dates,
            "history_prices": history_prices
        }
    
    # Generate visualization
    print("\nGenerating visualization...")
    recent_data = data.iloc[-60:].copy()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Main price chart
    ax1.plot(recent_data.index, recent_data['Close'], 
             label='Actual Price', color='#2E86AB', linewidth=2, marker='o', markersize=3)
    
    # Plot prediction
    ax1.plot([last_date, next_date], [current_close, predicted_price], 
             'r--', linewidth=2, label='Prediction')
    ax1.plot(next_date, predicted_price, 'r*', markersize=15, 
             label=f'Predicted: {predicted_price:.4f}')
    
    # Add Bollinger Bands if available
    if 'BB_Upper' in recent_data.columns:
        ax1.fill_between(recent_data.index, 
                         recent_data['BB_Upper'], 
                         recent_data['BB_Lower'], 
                         alpha=0.2, color='gray', label='Bollinger Bands')
    
    ax1.set_title(f'EUR/USD Price Prediction for {next_date.strftime("%Y-%m-%d")}', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Price', fontsize=12)
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Technical indicators subplot
    ax2_twin = ax2.twinx()
    
    # RSI
    if 'RSI' in recent_data.columns:
        ax2.plot(recent_data.index, recent_data['RSI'], 
                label='RSI', color='purple', linewidth=1.5)
        ax2.axhline(y=70, color='r', linestyle='--', alpha=0.5, linewidth=1)
        ax2.axhline(y=30, color='g', linestyle='--', alpha=0.5, linewidth=1)
        ax2.set_ylabel('RSI', fontsize=12, color='purple')
        ax2.tick_params(axis='y', labelcolor='purple')
        ax2.set_ylim(0, 100)
    
    # MACD
    if 'MACD' in recent_data.columns and 'MACD_Signal' in recent_data.columns:
        ax2_twin.plot(recent_data.index, recent_data['MACD'], 
                     label='MACD', color='#E63946', linewidth=1.5)
        ax2_twin.plot(recent_data.index, recent_data['MACD_Signal'], 
                     label='Signal', color='#06FFA5', linewidth=1.5)
        ax2_twin.set_ylabel('MACD', fontsize=12, color='#E63946')
        ax2_twin.tick_params(axis='y', labelcolor='#E63946')
    
    ax2.set_title('Technical Indicators', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Combine legends
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='best')
    
    plt.tight_layout()
    
    static_dir = os.path.join(project_root, "src", "web", "static")
    os.makedirs(static_dir, exist_ok=True)
    output_file = os.path.join(static_dir, "prediction_chart.png")
    plt.savefig(output_file, dpi=100)
    print(f"   [OK] Chart saved to {output_file}")
    
    plt.close()
    
    print("\n[OK] Prediction complete!")
    return None


if __name__ == "__main__":
    predict_next_day()
