# EUR/USD Price Prediction ML Project

A machine learning web application that predicts the next day's closing price for the EUR/USD currency pair using historical data and LSTM neural networks.

> [!WARNING]
> This project is for **educational purposes only** and should not be used for actual financial trading decisions.

## Features

- **Data Fetching**: Automatically fetches 5 years of EUR/USD historical data using Yahoo Finance API
- **LSTM Model**: Uses a Long Short-Term Memory neural network for time series prediction
- **Web Interface**: Interactive Flask-based web application with real-time predictions
- **Visualization**: Displays prediction charts and historical data analysis
- **RESTful API**: JSON API endpoint for programmatic access to predictions

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
    git clone <repository-url>
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
 
 You can run all components of the project using the `main.py` script.
 
 ### Step 1: Fetch Data
 
 Download the latest EUR/USD historical data:
 
 ```bash
 python main.py --fetch
 ```
 
 This will download 5 years of data and save it to `data/eurusd_data.csv`.
 
 ### Step 2: Train the Model
 
 Train the LSTM model on the fetched data:
 
 ```bash
 python main.py --train
 ```
 
 This will:
 - Load and preprocess the data
 - Train the LSTM neural network
 - Save the trained model to `models/eurusd_lstm_model.h5`
 - Display training metrics and visualizations
 
 ### Step 3: Run the Web Application
 
 Start the Flask web server:
 
 ```bash
 python main.py --web
 ```
 
 Then open your browser and navigate to:
 ```
 http://127.0.0.1:5000
 ```
 
 ### Make a Single Prediction
 
 To generate a prediction for the next trading day without starting the web app:
 
 ```bash
 python main.py --predict
 ```

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
- [ ] Add technical indicators (RSI, MACD, etc.)
- [ ] Deploy to cloud platform (Heroku, AWS, etc.)
- [ ] Add user authentication
- [ ] Historical prediction accuracy dashboard
- [ ] Real-time data updates

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

This project is for educational purposes only. Please ensure compliance with data provider terms of service and local financial regulations.

## Disclaimer

This application is intended for educational and research purposes only. Predictions made by this model should not be used as financial advice. Cryptocurrency and forex trading carries significant risk. Always consult with a qualified financial advisor before making investment decisions.
