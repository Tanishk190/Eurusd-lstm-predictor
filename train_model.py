import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import os

def calculate_rsi(data, window=14):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def train_model():
    print("Loading data...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_file = os.path.join(script_dir, "data", "eurusd_data.csv")
    
    try:
        df = pd.read_csv(data_file, index_col=0, parse_dates=True, header=[0, 1])
    except FileNotFoundError:
        print("Data file not found. Run fetch_data.py first.")
        return
    except Exception as e:
        # Fallback for simpler CSV structure
        df = pd.read_csv(data_file, index_col=0, parse_dates=True)

    # Handle MultiIndex columns if present (common in new yfinance)
    if isinstance(df.columns, pd.MultiIndex):
        try:
            # Try to extract the specific ticker level
            df = df.xs('EURUSD=X', level=1, axis=1)
        except:
            # If that fails, maybe the columns are just Price, Ticker but flattened or different
            # Let's just try to find 'Close'
            pass
            
    # Flatten columns if they are still MultiIndex but we couldn't select by ticker
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    data = pd.DataFrame(index=df.index)
    
    if 'Close' in df.columns:
        data['Close'] = df['Close']
    elif 'Adj Close' in df.columns:
        data['Close'] = df['Adj Close']
    else:
        print("Could not find 'Close' column. Columns are:", df.columns)
        return

    # Feature Engineering
    print("Generating features...")
    data['SMA_5'] = data['Close'].rolling(window=5).mean()
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['RSI'] = calculate_rsi(data['Close'])
    
    # Lagged features
    data['Lag_1'] = data['Close'].shift(1)
    data['Lag_2'] = data['Close'].shift(2)
    
    # Target: Tomorrow's Close
    data['Target'] = data['Close'].shift(-1)
    
    # Drop NaNs
    data = data.dropna()
    
    features = ['Close', 'SMA_5', 'SMA_20', 'SMA_50', 'RSI', 'Lag_1', 'Lag_2']
    X = data[features]
    y = data['Target']
    
    # Split data (Time-series split)
    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]
    
    print(f"Training Random Forest on {len(X_train)} samples...")
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Evaluate
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    
    print(f"Model Evaluation:")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    
    # Save model to models directory
    models_dir = os.path.join(script_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    model_file = os.path.join(models_dir, "rf_model.pkl")
    
    joblib.dump(model, model_file)
    print(f"Model saved to {model_file}")

if __name__ == "__main__":
    train_model()
