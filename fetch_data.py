import yfinance as yf
import pandas as pd
import os

def fetch_data():
    print("Fetching EUR/USD data...")
    # Fetch data for the last 5 years
    data = yf.download("EURUSD=X", period="5y")
    
    if data.empty:
        print("No data found. Please check your internet connection or the ticker symbol.")
        return

    # Save to CSV in data directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    output_file = os.path.join(data_dir, "eurusd_data.csv")
    data.to_csv(output_file)
    print(f"Data saved to {output_file}")
    print(data.tail())

if __name__ == "__main__":
    fetch_data()
