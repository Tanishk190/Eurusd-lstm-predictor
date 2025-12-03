import argparse
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    parser = argparse.ArgumentParser(description="EUR/USD Prediction Project Entry Point")
    
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument('-f', '--fetch', action='store_true', help='Fetch latest data')
    group.add_argument('-t', '--train', action='store_true', help='Train the LSTM model')
    group.add_argument('-p', '--predict', action='store_true', help='Make a prediction for the next day')
    group.add_argument('-w', '--web', action='store_true', help='Run the web application')
    group.add_argument('--pipeline', action='store_true', help='Run full pipeline: Fetch -> Train -> Web')
    group.add_argument('-i', '--interactive', action='store_true', help='Show interactive menu')
    
    args = parser.parse_args()
    
    # Interactive mode if requested OR no arguments provided
    if args.interactive or not any([args.fetch, args.train, args.predict, args.web, args.pipeline]):
        print("\nWelcome to EUR/USD Predictor!")
        print("Please select an action:")
        print("1. Fetch latest data")
        print("2. Train model")
        print("3. Make prediction")
        print("4. Run web application")
        print("5. Run full pipeline (Fetch -> Train -> Web)")
        print("6. Exit")
        
        try:
            choice = input("\nEnter choice (1-6): ")
            if choice == '1':
                args.fetch = True
            elif choice == '2':
                args.train = True
            elif choice == '3':
                args.predict = True
            elif choice == '4':
                args.web = True
            elif choice == '5':
                args.pipeline = True
            else:
                print("Exiting...")
                return
        except KeyboardInterrupt:
            print("\nExiting...")
            return
    
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
        app.run(debug=True, host='0.0.0.0', port=5000)
        
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
        app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)

if __name__ == "__main__":
    main()
