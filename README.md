# EUR/USD Price Prediction ML Project

A machine learning web application that predicts the next day's closing price for the EUR/USD currency pair using historical data and LSTM neural networks.

> [!WARNING]
> This project is for **educational purposes only** and should not be used for actual financial trading decisions.

## Features

- **Data Fetching**: Automatically fetches 5 years of EUR/USD historical data using Yahoo Finance API
- **LSTM Model**: Uses a Long Short-Term Memory neural network for time series prediction
- **Web Interface**: Interactive Flask-based web application with real-time predictions
- **Visualization**: Displays prediction charts and historical/backtest analysis
- **Backtesting**: One-click historical backtest with MAE/MSE/RMSE and chart
- **RESTful API**: JSON API endpoints for programmatic access to predictions and backtests

## Project Structure
 
 ```
 ML-Project/
 ├── data/                   # Data files
 ├── models/                 # Saved model artifacts
 ├── src/                    # Source code
 │   ├── data/               # Data fetching
 │   ├── features/           # Feature engineering
 │   ├── models/             # Training and prediction
 │   └── web/                # Web application
 ├── main.py                 # Unified entry point
 └── ...
 ```
 
 ## Installation
 
 ### Prerequisites
 
 - Python 3.7+
 - pip (Python package manager)
 
 ### Setup
 
 1. **Clone the repository**
    ```bash
    git clone https://github.com/Tanishk190/Eurusd-lstm-predictor.git
    cd ML-Project
    ```
 
 2. **Create a virtual environment** (recommended)
    ```bash
    python -m venv venv
    
    # On Windows
    venv\Scripts\activate
    
    # On macOS/Linux
    source venv/bin/activate
    ```
 
 3. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```
 
 ## Usage
 
 You can run the project using the unified `main.py` script.
 
 ### Interactive Menu (Recommended)
 
 Simply run the script without arguments to see the interactive menu:
 
 ```bash
 python main.py
 ```
 
 You will be prompted to choose an action:
 1. Fetch latest data
 2. Train model
 3. Make prediction
 4. Run web application
4. Backtest model on historical data
5. Run web application
6. Run full pipeline
7. Exit
 
 ### Command Line Arguments
 
 You can also use command-line flags for automation:
 
 | Flag | Alias | Description |
 |------|-------|-------------|
 | `--fetch` | `-f` | Fetch latest data |
 | `--train` | `-t` | Train the LSTM model |
 | `--predict` | `-p` | Make a prediction for the next day |
| `--backtest` | `-b` | Run historical backtest with metrics and chart |
 | `--web` | `-w` | Run the web application |
 | `--pipeline` | | Run full pipeline (Fetch -> Train -> Web) |
 | `--interactive` | `-i` | Show interactive menu |
 
 Example:
 ```bash
 python main.py -p  # Make a prediction
 python main.py --pipeline # Run everything
python main.py --backtest # Run historical backtest
 ```

 ### Network Access (Run on other devices)
 
 The web application is configured to be accessible from other devices on the same network.
 
 1. Run the web app: `python main.py -w`
 2. Find your computer's local IP address (e.g., `192.168.x.x`).
 3. On your phone or other device, navigate to `http://YOUR_LOCAL_IP:5000`.
 
 > **Note:** You may need to allow port 5000 through your firewall.

 ### Model Details
 
 - **Precision**: Predictions are displayed with **5 decimal places** (e.g., 1.08567).
 - **Reproducibility**: Random seeds are fixed to ensure consistent results across training runs.

### API Endpoint

Get predictions programmatically:

```bash
curl http://127.0.0.1:5000/api/predict
```

Response format:
```json
{
  "status": "success",
  "data": {
    "prediction": 1.0856,
    "last_price": 1.0850,
    "change": 0.0006,
    "timestamp": "2025-12-03"
  }
}
```

### Backtest Endpoint

Run a backtest and get metrics:

```bash
curl http://127.0.0.1:5000/api/backtest
```

Returns MSE, MAE, RMSE, and test sample count. A chart is also saved locally at `src/web/static/backtest_results.png` after a run (ignored by git; generated on demand).

## How It Works

1. **Data Collection**: `fetch_data.py` retrieves historical EUR/USD exchange rates from Yahoo Finance
2. **Preprocessing**: Data is normalized and split into sequences for time series analysis
3. **Model Training**: An LSTM neural network learns patterns from historical price movements
4. **Prediction**: The trained model predicts the next day's closing price based on recent trends
5. **Visualization**: Results are displayed through a web interface with charts and metrics

## Model Architecture

- **Input Layer**: 60 days of historical prices
- **LSTM Layers**: Two stacked LSTM layers (50 units each)
- **Dropout**: 20% dropout for regularization
- **Output Layer**: Single dense layer for next-day prediction
- **Optimizer**: Adam
- **Loss Function**: Mean Squared Error (MSE)

## Screenshots (generated locally)

These images are generated when you run the app and are ignored by git. If you don’t see them, run the corresponding action first.

- Prediction chart (after running a prediction):  
  `src/web/static/prediction_chart.png`

- Backtest chart (after running a backtest):  
  `src/web/static/backtest_results.png`

## Dependencies

- **Flask**: Web framework for the application
- **yfinance**: Fetches financial data from Yahoo Finance
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **tensorflow/keras**: Deep learning framework for LSTM model
- **scikit-learn**: Data preprocessing and scaling
- **matplotlib**: Visualization and chart generation

## Future Enhancements

- [ ] Add multiple currency pair support
- [ ] Implement model performance tracking
- [✓] Add technical indicators (RSI, MACD, etc.)
- [ ] Deploy to cloud platform (Heroku, AWS, etc.)
- [ ] Add user authentication
- [ ] Historical prediction accuracy dashboard
- [ ] Real-time data updates

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

This project is for educational purposes only. Please ensure compliance with data provider terms of service and local financial regulations.

## Disclaimer

This application is intended for educational and research purposes only. Predictions made by this model should not be used as financial advice. Cryptocurrency and forex trading carries significant risk. Always consult with a qualified financial advisor before making investment decisions.
