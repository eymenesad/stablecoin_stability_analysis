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

def get_usdt_reference(raw_dir):
    """
    Loads USDT/USD data to serve as a reference for normalizing USDT pairs.
    Prefers Coinbase, falls back to Kraken.
    """
    coinbase_path = os.path.join(raw_dir, 'USDT_USD_coinbase.csv')
    kraken_path = os.path.join(raw_dir, 'USDT_USD_kraken.csv')
    
    if os.path.exists(coinbase_path):
        return load_data(coinbase_path)
    elif os.path.exists(kraken_path):
        return load_data(kraken_path)
    else:
        print("Warning: No USDT/USD data found. USDT pairs will not be normalized.")
        return None

def normalize_to_usd(df, symbol_name, usdt_df):
    """
    If the symbol is XXX/USDT, converts it to XXX/USD using the USDT/USD reference.
    """
    if 'USDT' in symbol_name and 'USD' not in symbol_name.split('_')[0]: # e.g. BUSD_USDT
        # This is a pair against USDT (e.g. BUSD/USDT)
        # We need to merge with USDT/USD on timestamp
        
        # Rename USDT columns
        usdt_subset = usdt_df[['timestamp', 'close']].rename(columns={'close': 'usdt_price'})
        
        # Merge
        merged = pd.merge(df, usdt_subset, on='timestamp', how='inner')
        
        # Calculate USD price
        merged['close_raw'] = merged['close']
        merged['close'] = merged['close'] * merged['usdt_price']
        
        # Drop the helper column
        merged = merged.drop(columns=['usdt_price'])
        
        print(f"  Normalized {symbol_name} to USD using USDT reference. Records: {len(df)} -> {len(merged)}")
        return merged
    else:
        return df

def calculate_features(df, symbol_name):
    # 1. Price Deviation (Target is $1.00)
    df['deviation_abs'] = (df['close'] - 1.0).abs()
    df['deviation_pct'] = (df['close'] - 1.0) / 1.0
    
    # 2. Volatility (Rolling Standard Deviation)
    df['volatility_24h'] = df['close'].rolling(window=24).std()
    
    # 3. Time Features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    # 4. Momentum / Price Change
    df['price_change_1h'] = df['close'].pct_change(1)
    df['price_change_6h'] = df['close'].pct_change(6)
    df['price_change_24h'] = df['close'].pct_change(24)
    
    # 5. Volume Features
    df['volume_ma_24h'] = df['volume'].rolling(window=24).mean()
    df['volume_std_24h'] = df['volume'].rolling(window=24).std()
    
    # 6. Lag Features (Explicitly for forecasting)
    # We want the model to see "what happened 1h ago, 2h ago" etc.
    # The current row 'price_change_1h' represents change from t-1 to t.
    # We add explicit lags of key features
    for lag in [1, 2, 3, 6, 12, 24]:
        df[f'deviation_lag_{lag}h'] = df['deviation_pct'].shift(lag)
        df[f'volatility_lag_{lag}h'] = df['volatility_24h'].shift(lag)
        df[f'volume_lag_{lag}h'] = df['volume'].shift(lag)
        
    # 7. Advanced Features (Velocity & Z-Scores)
    # Velocity (First Difference)
    df['deviation_velocity'] = df['deviation_pct'].diff()
    df['volume_velocity'] = df['volume'].pct_change()
    
    # Rolling Z-Scores (24h)
    # (Value - Mean) / Std
    df['deviation_zscore_24h'] = (df['deviation_pct'] - df['deviation_pct'].rolling(24).mean()) / df['deviation_pct'].rolling(24).std()
    df['volume_zscore_24h'] = (df['volume'] - df['volume_ma_24h']) / df['volume_std_24h']
    
    # 8. FORECASTING TARGETS
    # We want to predict stability/deviation in the FUTURE.
    # Target: Deviation 1 hour from now.
    df['target_close_1h'] = df['close'].shift(-1)
    df['target_deviation_pct_1h'] = (df['target_close_1h'] - 1.0) / 1.0
    
    # Target: Deviation 6 hours from now
    df['target_close_6h'] = df['close'].shift(-6)
    df['target_deviation_pct_6h'] = (df['target_close_6h'] - 1.0) / 1.0
    
    # Labeling based on FUTURE deviation (1h)
    def get_label(dev_pct):
        if pd.isna(dev_pct):
            return None
        dev = abs(dev_pct)
        if dev < 0.001: # 0.1%
            return 'High'
        elif dev < 0.01: # 1.0%
            return 'Medium'
        else:
            return 'Low'
            
    df['target_label_1h'] = df['target_deviation_pct_1h'].apply(get_label)
    
    return df

def add_market_context(target_df, btc_df, eth_df):
    """
    Adds BTC and ETH price/volatility as features to the target stablecoin dataframe.
    """
    btc_subset = btc_df[['timestamp', 'close', 'volume']].rename(columns={'close': 'btc_close', 'volume': 'btc_volume'})
    eth_subset = eth_df[['timestamp', 'close', 'volume']].rename(columns={'close': 'eth_close', 'volume': 'eth_volume'})
    
    merged = pd.merge(target_df, btc_subset, on='timestamp', how='left')
    merged = pd.merge(merged, eth_subset, on='timestamp', how='left')
    
    # Calculate correlations (24h rolling)
    merged['corr_btc_24h'] = merged['close'].rolling(window=24).corr(merged['btc_close'])
    merged['corr_eth_24h'] = merged['close'].rolling(window=24).corr(merged['eth_close'])
    
    # Market Returns
    merged['btc_return_24h'] = merged['btc_close'].pct_change(24)
    merged['eth_return_24h'] = merged['eth_close'].pct_change(24)
    
    return merged

def main():
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)
        
    # Load Market Context
    btc_path_coinbase = os.path.join(RAW_DIR, 'BTC_USD_coinbase.csv')
    eth_path_coinbase = os.path.join(RAW_DIR, 'ETH_USD_coinbase.csv')
    # Fallbacks
    btc_path = btc_path_coinbase if os.path.exists(btc_path_coinbase) else os.path.join(RAW_DIR, 'BTC_USD_kraken.csv')
    eth_path = eth_path_coinbase if os.path.exists(eth_path_coinbase) else os.path.join(RAW_DIR, 'ETH_USD_kraken.csv')
    
    btc_df = load_data(btc_path) if os.path.exists(btc_path) else None
    eth_df = load_data(eth_path) if os.path.exists(eth_path) else None
    
    # Load USDT Reference for Normalization
    usdt_ref_df = get_usdt_reference(RAW_DIR)
    
    # Process all files
    files = glob.glob(os.path.join(RAW_DIR, '*.csv'))
    for filepath in files:
        filename = os.path.basename(filepath)
        if 'BTC' in filename or 'ETH' in filename:
            continue
            
        print(f"Processing {filename}...")
        df = load_data(filepath)
        
        # Normalize USDT pairs
        if usdt_ref_df is not None:
            df = normalize_to_usd(df, filename, usdt_ref_df)
            
        df = calculate_features(df, filename)
        
        if btc_df is not None and eth_df is not None:
            df = add_market_context(df, btc_df, eth_df)
            
        # Drop rows with NaNs (due to rolling windows or future targets)
        # We drop NaNs at the END of the pipeline to ensure we have clean data for modeling
        # However, for 'future' targets, the last N rows will be NaN. We can keep them or drop them.
        # Let's drop them to avoid issues in modeling.
        df = df.dropna()
            
        save_path = os.path.join(PROCESSED_DIR, f"processed_{filename}")
        df.to_csv(save_path, index=False)
        print(f"Saved processed data to {save_path}")

if __name__ == "__main__":
    main()
