import pandas as pd
import numpy as np
import os
import glob

RAW_DIR = 'data/raw'
PROCESSED_DIR = 'data/processed'

def load_data(filepath):
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df

def calculate_features(df, symbol_name):
    # 1. Price Deviation (assuming target is $1.00)
    # Note: For pairs like BUSD/USDT, we are measuring deviation from USDT, which is a proxy for USD.
    df['deviation_abs'] = (df['close'] - 1.0).abs()
    df['deviation_pct'] = (df['close'] - 1.0) / 1.0
    
    # 2. Volatility (Rolling Standard Deviation)
    # 24h rolling window
    df['volatility_24h'] = df['close'].rolling(window=24).std()
    
    # 3. Time Features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    # 4. Labeling
    # High Stability: < 0.1% deviation
    # Medium Stability: 0.1% - 1.0%
    # Low Stability: > 1.0%
    # We use the absolute percentage deviation for labeling
    
    def get_label(row):
        dev = abs(row['deviation_pct'])
        if dev < 0.001: # 0.1%
            return 'High'
        elif dev < 0.01: # 1.0%
            return 'Medium'
        else:
            return 'Low'
            
    df['stability_label'] = df.apply(get_label, axis=1)
    
    return df

def add_market_context(target_df, btc_df, eth_df):
    """
    Adds BTC and ETH price/volatility as features to the target stablecoin dataframe.
    """
    # Merge on timestamp
    # Rename BTC/ETH columns to avoid collision
    btc_subset = btc_df[['timestamp', 'close', 'volume']].rename(columns={'close': 'btc_close', 'volume': 'btc_volume'})
    eth_subset = eth_df[['timestamp', 'close', 'volume']].rename(columns={'close': 'eth_close', 'volume': 'eth_volume'})
    
    merged = pd.merge(target_df, btc_subset, on='timestamp', how='left')
    merged = pd.merge(merged, eth_subset, on='timestamp', how='left')
    
    # Calculate correlations (e.g. 24h rolling correlation with BTC)
    merged['corr_btc_24h'] = merged['close'].rolling(window=24).corr(merged['btc_close'])
    merged['corr_eth_24h'] = merged['close'].rolling(window=24).corr(merged['eth_close'])
    
    return merged

def main():
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)
        
    # Load Market Context (BTC/ETH) first
    btc_path = os.path.join(RAW_DIR, 'BTC_USD_kraken.csv')
    eth_path = os.path.join(RAW_DIR, 'ETH_USD_kraken.csv')
    
    btc_df = load_data(btc_path) if os.path.exists(btc_path) else None
    eth_df = load_data(eth_path) if os.path.exists(eth_path) else None
    
    if btc_df is None or eth_df is None:
        print("Warning: BTC or ETH data not found. Correlation features will be skipped.")
    
    # Process all other files
    files = glob.glob(os.path.join(RAW_DIR, '*.csv'))
    for filepath in files:
        filename = os.path.basename(filepath)
        if 'BTC' in filename or 'ETH' in filename:
            continue
            
        print(f"Processing {filename}...")
        df = load_data(filepath)
        df = calculate_features(df, filename)
        
        if btc_df is not None and eth_df is not None:
            df = add_market_context(df, btc_df, eth_df)
            
        save_path = os.path.join(PROCESSED_DIR, f"processed_{filename}")
        df.to_csv(save_path, index=False)
        print(f"Saved processed data to {save_path}")

if __name__ == "__main__":
    main()
