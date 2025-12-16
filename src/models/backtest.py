"""
Backtesting Script for EUR/USD LSTM Model

Uses the trained LSTM model, saved scaler, and feature configuration
to generate walk-forward predictions on historical data and evaluate
out-of-sample performance.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from tensorflow import keras

from src.features.indicators import add_all_indicators
from src.models.train_model import create_sequences


def backtest_model(test_split: float = 0.2, seq_length: int = 60, save_plots: bool = True):
    """
    Backtest the saved LSTM model on historical EUR/USD data.

    This function:
    - Loads the historical CSV data
    - Recomputes all technical indicators
    - Loads the saved scaler and feature configuration
    - Builds sequences and runs the trained model in a walk-forward way
    - Evaluates metrics on the held‑out test portion
    - Optionally saves a plot of predictions vs actual prices

    Args:
        test_split: Fraction of data (in sequences) reserved for testing (default: 0.2)
        seq_length: Lookback window used during training (default: 60)
        save_plots: If True, saves a PNG chart with backtest results
    """
    print("=" * 60)
    print("LSTM Model Backtest on Historical EUR/USD Data")
    print("=" * 60)

    # Locate project directories
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    data_file = os.path.join(project_root, "data", "eurusd_data.csv")
    models_dir = os.path.join(project_root, "models")

    # ------------------------------------------------------------------
    # 1) Load historical data
    # ------------------------------------------------------------------
    print("\n[1/6] Loading historical data...")
    if not os.path.exists(data_file):
        print(f"   [ERROR] Data file not found at: {data_file}")
        print("   Please fetch data first with: python main.py --fetch")
        return None

    try:
        df = pd.read_csv(data_file, index_col=0, parse_dates=True, header=[0, 1])
    except Exception:
        df = pd.read_csv(data_file, index_col=0, parse_dates=True)

    # Handle MultiIndex columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        try:
            df = df.xs("EURUSD=X", level=1, axis=1)
        except Exception:
            pass

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    print(f"   Loaded {len(df)} rows from CSV")

    # ------------------------------------------------------------------
    # 2) Add indicators and clean data
    # ------------------------------------------------------------------
    print("\n[2/6] Calculating technical indicators...")
    data = add_all_indicators(df)
    data = data.dropna()
    print(f"   After indicators & NaN removal: {len(data)} samples")

    # ------------------------------------------------------------------
    # 3) Load scaler and feature configuration
    # ------------------------------------------------------------------
    print("\n[3/6] Loading scaler and feature configuration...")
    scaler_file = os.path.join(models_dir, "scaler.pkl")
    feature_file = os.path.join(models_dir, "feature_columns.pkl")
    model_file = os.path.join(models_dir, "eurusd_lstm_model.h5")

    if not os.path.exists(model_file):
        print(f"   [ERROR] Trained model not found at: {model_file}")
        print("   Please train the model first with: python main.py --train")
        return None

    try:
        scaler = joblib.load(scaler_file)
        feature_columns = joblib.load(feature_file)
    except Exception as e:
        print(f"   [ERROR] Could not load scaler or feature list: {e}")
        return None

    # Filter to only use available features (same as training)
    available_features = [col for col in feature_columns if col in data.columns]
    if not available_features:
        print("   [ERROR] None of the saved feature columns are present in the data.")
        return None

    print(f"   Using {len(available_features)} features for backtest")
    dataset = data[available_features].values

    # ------------------------------------------------------------------
    # 4) Scale data and create sequences
    # ------------------------------------------------------------------
    print("\n[4/6] Scaling data and creating sequences...")
    try:
        scaled_data = scaler.transform(dataset)
    except Exception as e:
        print(f"   [ERROR] Failed to apply scaler: {e}")
        return None

    if len(scaled_data) <= seq_length:
        print(f"   [ERROR] Not enough data for seq_length={seq_length}")
        return None

    X, y = create_sequences(scaled_data, seq_length=seq_length)
    print(f"   Created {len(X)} sequences")

    # Split into train/test based on chronological order
    split_idx = int((1.0 - test_split) * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"   Train sequences: {len(X_train)}")
    print(f"   Test sequences:  {len(X_test)}")

    if len(X_test) == 0:
        print("   [ERROR] Test split resulted in zero test samples.")
        return None

    # ------------------------------------------------------------------
    # 5) Load model and generate predictions
    # ------------------------------------------------------------------
    print("\n[5/6] Loading model and generating predictions...")
    try:
        model = keras.models.load_model(model_file)
    except Exception as e:
        print(f"   [ERROR] Failed to load model: {e}")
        return None

    # Predict only on the test set for true out-of-sample evaluation
    test_predictions_scaled = model.predict(X_test, verbose=0)

    # Inverse transform predictions and targets (Close price is first column)
    test_pred_full = np.zeros((len(test_predictions_scaled), scaled_data.shape[1]))
    test_pred_full[:, 0] = test_predictions_scaled.flatten()
    test_pred_actual = scaler.inverse_transform(test_pred_full)[:, 0]

    y_test_full = np.zeros((len(y_test), scaled_data.shape[1]))
    y_test_full[:, 0] = y_test
    y_test_actual = scaler.inverse_transform(y_test_full)[:, 0]

    # Align dates with test sequences (last date of each input window)
    all_dates = data.index[seq_length:]  # Dates corresponding to y values
    test_dates = all_dates[split_idx:]

    # ------------------------------------------------------------------
    # 6) Compute metrics and optionally plot
    # ------------------------------------------------------------------
    print("\n[6/6] Computing backtest metrics...")
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    mse = mean_squared_error(y_test_actual, test_pred_actual)
    mae = mean_absolute_error(y_test_actual, test_pred_actual)
    rmse = np.sqrt(mse)

    print("\n" + "=" * 60)
    print("Backtest Performance (Out-of-Sample Test Set)")
    print("=" * 60)
    print(f"Test Samples: {len(y_test_actual)}")
    print(f"MSE:  {mse:.6f}")
    print(f"MAE:  {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print("=" * 60)

    if save_plots:
        print("\nGenerating backtest plot...")
        fig, ax = plt.subplots(figsize=(14, 6))

        ax.plot(test_dates, y_test_actual, label="Actual Close", color="#2E86AB", linewidth=2)
        ax.plot(test_dates, test_pred_actual, label="Predicted Close", color="#E63946", linewidth=2, alpha=0.8)

        ax.set_title("EUR/USD Backtest - Actual vs Predicted Close (Test Period)", fontsize=14, fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("EUR/USD Price")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        static_dir = os.path.join(project_root, "src", "web", "static")
        os.makedirs(static_dir, exist_ok=True)
        output_file = os.path.join(static_dir, "backtest_results.png")
        plt.savefig(output_file, dpi=100)
        plt.close(fig)

        print(f"   [OK] Backtest chart saved to {output_file}")

    print("\n[OK] Backtest complete!")

    return {
        "mse": float(mse),
        "mae": float(mae),
        "rmse": float(rmse),
        "n_test_samples": int(len(y_test_actual)),
    }


if __name__ == "__main__":
    backtest_model()


