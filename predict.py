import pandas as pd
import joblib
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import os

def calculate_rsi(series, window=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def predict_next_day(return_data=False):
    print("Fetching latest data for prediction...")
    # Fetch enough data to calculate SMA_50 and RSI
    df = yf.download("EURUSD=X", period="6mo")
    
    # Handle MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.xs('EURUSD=X', level=1, axis=1)
        except:
            pass
            
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    data = pd.DataFrame(index=df.index)
    if 'Close' in df.columns:
        data['Close'] = df['Close']
    elif 'Adj Close' in df.columns:
        data['Close'] = df['Adj Close']
    else:
        print("Could not find 'Close' column.")
        return None

    # Re-create features
    data['SMA_5'] = data['Close'].rolling(window=5).mean()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['RSI'] = calculate_rsi(data['Close'])
    data['Lag_1'] = data['Close'].shift(1)
    data['Lag_2'] = data['Close'].shift(2)
    
    # We use the LAST available row to predict the NEXT day
    last_row = data.iloc[-1:]
    
    if last_row.isnull().values.any():
        print("Not enough data to calculate features for the latest date.")
        return None

    features = ['Close', 'SMA_5', 'SMA_20', 'SMA_50', 'RSI', 'Lag_1', 'Lag_2']
    X_new = last_row[features]
    
    print("Loading model...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_file = os.path.join(script_dir, "models", "rf_model.pkl")
    
    try:
        model = joblib.load(model_file)
    except FileNotFoundError:
        print("Model not found. Train the model first.")
        return None
        
    prediction = model.predict(X_new)
    predicted_price = prediction[0]
    
    last_date = last_row.index[0]
    # If last_date is a Timestamp, we can add Timedelta. 
    # Note: yfinance usually returns timezone-aware datetimes.
    next_date = last_date + pd.Timedelta(days=1)
    # Adjust for weekends if needed (simple approach: just add 1 day for now)
    if next_date.weekday() >= 5: # Saturday or Sunday
        next_date += pd.Timedelta(days=(7 - next_date.weekday()))

    print(f"\nDate: {last_date.date()}")
    print(f"Current Close: {last_row['Close'].values[0]:.4f}")
    print(f"Predicted Price for Next Trading Day ({next_date.date()}): {predicted_price:.4f}")

    if return_data:
        # Get last 30 days for context
        recent_data = data.iloc[-30:].copy()
        history_dates = [d.strftime('%Y-%m-%d') for d in recent_data.index]
        history_prices = recent_data['Close'].tolist()
        
        return {
            "current_date": last_date.strftime('%Y-%m-%d'),
            "current_close": float(last_row['Close'].values[0]),
            "predicted_date": next_date.strftime('%Y-%m-%d'),
            "predicted_price": float(predicted_price),
            "history_dates": history_dates,
            "history_prices": history_prices
        }

    # --- Visualization ---
    print("Generating chart...")
    # Get last 30 days for context
    recent_data = data.iloc[-30:].copy()
    
    plt.figure(figsize=(12, 6))
    plt.plot(recent_data.index, recent_data['Close'], label='Actual History', marker='o', markersize=4)
    
    # Plot the connection from last actual to predicted
    plt.plot([last_date, next_date], [recent_data['Close'].iloc[-1], predicted_price], 'r--', label='Prediction Path')
    plt.plot(next_date, predicted_price, 'rx', markersize=10, markeredgewidth=2, label=f'Predicted: {predicted_price:.4f}')
    
    plt.title(f'EUR/USD Price Prediction for {next_date.date()}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_file = os.path.join(script_dir, "static", "prediction_chart.png")
    plt.savefig(output_file)
    print(f"Chart saved to {output_file}")

if __name__ == "__main__":
    predict_next_day()
