import pandas as pd
import numpy as np
import os

def analyze_squid_ink(file_directory="."):
    results = []
    
    for day in [1, -1, 0]:
        file_name = f"prices_round_2_day_{day}.csv"
        file_path = os.path.join(file_directory, file_name)
        
        if not os.path.exists(file_path):
            print(f"ERROR: File not found at {os.path.abspath(file_path)}")
            print("Please ensure either:")
            print(f"1. Files are in current working directory ({os.getcwd()})")
            print(f"2. You specify the correct directory using analyze_squid_ink('your/directory/path')")
            results.append((day, None))
            continue

        try:
            # Read and process data
            df = pd.read_csv(file_path, delimiter=';')
            
            # Calculate mid-price and returns
            df['mid_price'] = (df['ask_price_1'] + df['bid_price_1']) / 2
            df['returns'] = df['mid_price'].pct_change().fillna(0)
            
            # Calculate Sharpe ratio
            sharpe = safe_sharpe(df['returns'])
            print(f"Day {day} Analysis Successful")
            print(f"  First price: {df['mid_price'].iloc[0]:.2f}")
            print(f"  Final price: {df['mid_price'].iloc[-1]:.2f}")
            print(f"  Sharpe Ratio: {sharpe:.2f}\n")
            results.append((day, sharpe))
            
        except Exception as e:
            print(f"ERROR processing {file_name}: {str(e)}")
            results.append((day, None))

    # Final summary
    print("\n=== Analysis Summary ===")
    for day, sharpe in results:
        if sharpe is not None:
            print(f"Day {day}: Sharpe Ratio = {sharpe:.2f}")
        else:
            print(f"Day {day}: Analysis failed (file missing or invalid data)")
    
    return results

def safe_sharpe(returns_series):
    # Clean data
    valid_returns = returns_series.replace([np.inf, -np.inf], np.nan).dropna()
    
    # Check calculation viability
    if len(valid_returns) < 2:
        return 0.0
    
    std_dev = valid_returns.std()
    if std_dev == 0:
        return 0.0
    
    # Annualize Sharpe ratio (252 trading days)
    return (valid_returns.mean() / std_dev) * np.sqrt(252)

if __name__ == "__main__":
    # First try current directory
    if any(os.path.exists(f"prices_round_2_day_{day}.csv") for day in [-1, 1, 0]):
        analyze_squid_ink()
    else:
        # If not found, prompt user
        print("CSV files not found in current directory.")
        user_path = input("Enter full path to directory containing files: ").strip()
        analyze_squid_ink(user_path)
