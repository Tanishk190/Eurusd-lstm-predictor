import argparse
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    parser = argparse.ArgumentParser(description="EUR/USD Prediction Project Entry Point")
    
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('--fetch', action='store_true', help='Fetch latest data')
    group.add_argument('--train', action='store_true', help='Train the LSTM model')
    group.add_argument('--predict', action='store_true', help='Make a prediction for the next day')
    group.add_argument('--web', action='store_true', help='Run the web application')
    group.add_argument('--pipeline', action='store_true', help='Run full pipeline: Fetch -> Train -> Web')
    
    args = parser.parse_args()
    
    # Default to pipeline if no arguments provided
    if not any([args.fetch, args.train, args.predict, args.web, args.pipeline]):
        print("No arguments provided. Defaulting to full pipeline...")
        args.pipeline = True
    
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
        
    elif args.pipeline:
        print("\n[1/3] STARTING PIPELINE: FETCH DATA")
        from src.data.fetch_data import fetch_data
        fetch_data()
        
        print("\n[2/3] STARTING PIPELINE: TRAIN MODEL")
        from src.models.train_model import train_lstm_model
        train_lstm_model()
        
        print("\n[3/3] STARTING PIPELINE: WEB APP")
        from src.web.app import app
        print("Starting web application...")
        # Disable reloader to prevent the script from running twice (fetching/training again)
        app.run(debug=True, use_reloader=False, port=5000)

if __name__ == "__main__":
    main()
