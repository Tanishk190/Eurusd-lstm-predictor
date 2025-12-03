"""
LSTM Model Training Script for EUR/USD Prediction
Uses technical indicators for feature engineering
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import joblib
import os
import random
from src.features.indicators import add_all_indicators


def set_seeds(seed=42):
    """Set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def create_sequences(data, seq_length=60):
    """
    Create sequences for LSTM training
    
    Args:
        data: Scaled feature array
        seq_length: Number of time steps to look back
    
    Returns:
        X: Input sequences, y: Target values
    """
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(data[i, 0])  # Predicting Close price (first column)
    return np.array(X), np.array(y)


def train_lstm_model():
    # Set seeds for reproducibility
    set_seeds(42)
    
    print("=" * 60)
    print("LSTM Model Training with Technical Indicators")
    print("=" * 60)
    
    # Load data
    print("\n[1/7] Loading data...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    data_file = os.path.join(project_root, "data", "eurusd_data.csv")
    
    try:
        df = pd.read_csv(data_file, index_col=0, parse_dates=True, header=[0, 1])
    except:
        df = pd.read_csv(data_file, index_col=0, parse_dates=True)
    
    # Handle MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.xs('EURUSD=X', level=1, axis=1)
        except:
            pass
    
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    print(f"   Loaded {len(df)} days of data")
    
    # Add all technical indicators
    print("\n[2/7] Calculating technical indicators...")
    data = add_all_indicators(df)
    print(f"   Generated {len(data.columns)} features")
    
    # Drop NaN values
    data = data.dropna()
    print(f"   {len(data)} samples after removing NaN values")
    
    # Select features for training
    feature_columns = [
        'Close', 'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50',
        'EMA_12', 'EMA_26', 'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
        'BB_Upper', 'BB_Middle', 'BB_Lower', 'BB_Bandwidth',
        'Momentum', 'Daily_Return', 'Log_Return'
    ]
    
    # Add ATR if available
    if 'ATR' in data.columns:
        feature_columns.append('ATR')
    
    # Filter to only use available features
    available_features = [col for col in feature_columns if col in data.columns]
    print(f"   Using {len(available_features)} features for training")
    
    dataset = data[available_features].values
    
    # Scale the data
    print("\n[3/7] Scaling features...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    
    # Save the scaler for later use in prediction
    models_dir = os.path.join(project_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    scaler_file = os.path.join(models_dir, "scaler.pkl")
    joblib.dump(scaler, scaler_file)
    print(f"   Scaler saved to {scaler_file}")
    
    # Save feature columns
    feature_file = os.path.join(models_dir, "feature_columns.pkl")
    joblib.dump(available_features, feature_file)
    print(f"   Feature list saved to {feature_file}")
    
    # Create sequences
    print("\n[4/7] Creating sequences...")
    seq_length = 60  # Use 60 days of history
    X, y = create_sequences(scaled_data, seq_length)
    
    print(f"   Created {len(X)} sequences")
    print(f"   Sequence shape: {X.shape}")
    
    # Split data (80-20 split for time series)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples: {len(X_test)}")
    
    # Build LSTM model
    print("\n[5/7] Building LSTM model...")
    model = Sequential([
        LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        
        LSTM(units=100, return_sequences=True),
        Dropout(0.2),
        
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        
        Dense(units=25),
        Dense(units=1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    
    print(f"   Model architecture:")
    model.summary()
    
    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.00001)
    
    # Train the model
    print("\n[6/7] Training LSTM model...")
    print("   This may take several minutes...")
    
    history = model.fit(
        X_train, y_train,
        batch_size=32,
        epochs=100,
        validation_data=(X_test, y_test),
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    # Evaluate the model
    print("\n[7/7] Evaluating model...")
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    
    # Inverse transform predictions (only the first column - Close price)
    train_pred_full = np.zeros((len(train_predictions), scaled_data.shape[1]))
    train_pred_full[:, 0] = train_predictions.flatten()
    train_pred_actual = scaler.inverse_transform(train_pred_full)[:, 0]
    
    test_pred_full = np.zeros((len(test_predictions), scaled_data.shape[1]))
    test_pred_full[:, 0] = test_predictions.flatten()
    test_pred_actual = scaler.inverse_transform(test_pred_full)[:, 0]
    
    # Get actual prices
    y_train_full = np.zeros((len(y_train), scaled_data.shape[1]))
    y_train_full[:, 0] = y_train
    y_train_actual = scaler.inverse_transform(y_train_full)[:, 0]
    
    y_test_full = np.zeros((len(y_test), scaled_data.shape[1]))
    y_test_full[:, 0] = y_test
    y_test_actual = scaler.inverse_transform(y_test_full)[:, 0]
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train_actual, train_pred_actual)
    train_mae = mean_absolute_error(y_train_actual, train_pred_actual)
    test_mse = mean_squared_error(y_test_actual, test_pred_actual)
    test_mae = mean_absolute_error(y_test_actual, test_pred_actual)
    
    print("\n" + "=" * 60)
    print("Model Performance Metrics")
    print("=" * 60)
    print(f"Training Set:")
    print(f"  MSE: {train_mse:.6f}")
    print(f"  MAE: {train_mae:.6f}")
    print(f"  RMSE: {np.sqrt(train_mse):.6f}")
    print(f"\nTest Set:")
    print(f"  MSE: {test_mse:.6f}")
    print(f"  MAE: {test_mae:.6f}")
    print(f"  RMSE: {np.sqrt(test_mse):.6f}")
    print("=" * 60)
    
    # Save the model
    model_file = os.path.join(models_dir, "eurusd_lstm_model.h5")
    model.save(model_file)
    print(f"\n[OK] Model saved to {model_file}")
    
    # Plot training history
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(y_test_actual[-100:], label='Actual Price', marker='o', markersize=3)
    plt.plot(test_pred_actual[-100:], label='Predicted Price', marker='x', markersize=3)
    plt.title('Last 100 Test Predictions vs Actual')
    plt.xlabel('Sample')
    plt.ylabel('EUR/USD Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    # Save to src/web/static
    static_dir = os.path.join(project_root, "src", "web", "static")
    os.makedirs(static_dir, exist_ok=True)
    chart_file = os.path.join(static_dir, "training_results.png")
    plt.savefig(chart_file)
    print(f"[OK] Training charts saved to {chart_file}")
    
    print("\n[OK] Training complete!")
    return model, history


if __name__ == "__main__":
    train_lstm_model()
