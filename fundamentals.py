import finagg
import pandas as pd
from datetime import datetime, timedelta

def get_fundamentals(ticker: str, years: int):
    print(f"Installing {ticker} fundamental data...")
    finagg.fundam.feat.fundam.install({ticker})

    try:
        # Get the start date based on the number of years specified
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)

        # Retrieve data from the refined source, falling back to the raw API if needed
        try:
            fundamentals = finagg.fundam.feat.fundam.from_refined(ticker, start=start_date.strftime("%Y-%m-%d"))
        except Exception:
            print(f"Refined data not found for {ticker}. Attempting to retrieve from raw API...")
            fundamentals = finagg.fundam.feat.fundam.from_api(ticker, start=start_date.strftime("%Y-%m-%d"))

        # Display the results
        print(fundamentals.head())

    except Exception as e:
        print(f"No data found for {ticker} from API or data might be unavailable.")
        print(f"Error: {e}")

# Run with specified ticker and years
if __name__ == '__main__':
    get_fundamentals(ticker="AAPL", years=5)
