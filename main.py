import argparse
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    parser = argparse.ArgumentParser(description="EUR/USD Prediction Project Entry Point")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--fetch', action='store_true', help='Fetch latest data')
    group.add_argument('--train', action='store_true', help='Train the LSTM model')
    group.add_argument('--predict', action='store_true', help='Make a prediction for the next day')
    group.add_argument('--web', action='store_true', help='Run the web application')
    
    args = parser.parse_args()
    
    if args.fetch:
        from src.data.fetch_data import fetch_data
        fetch_data()
        
    elif args.train:
        from src.models.train_model import train_lstm_model
        train_lstm_model()
        
    elif args.predict:
        from src.models.predict import predict_next_day
        predict_next_day()
        
    elif args.web:
        from src.web.app import app
        print("Starting web application...")
        app.run(debug=True, port=5000)

if __name__ == "__main__":
    main()
