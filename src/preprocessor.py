import pandas as pd
import numpy as np
import os
import glob

RAW_DIR = 'data/raw'
PROCESSED_DIR = 'data/processed'


def load_data(filepath):
    df = pd.read_csv(filepath)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df.sort_values('timestamp').reset_index(drop=True)


def get_usdt_reference(raw_dir):
    # prefer coinbase, fallback to kraken
    for exchange in ['coinbase', 'kraken']:
        path = os.path.join(raw_dir, f'USDT_USD_{exchange}.csv')
        if os.path.exists(path):
            return load_data(path)
    print("Warning: No USDT/USD reference found")
    return None


def normalize_to_usd(df, symbol_name, usdt_df):
    # convert XXX/USDT pairs to USD using USDT price
    if 'USDT' in symbol_name and 'USD' not in symbol_name.split('_')[0]:
        usdt_subset = usdt_df[['timestamp', 'close']].rename(columns={'close': 'usdt_price'})
        merged = pd.merge(df, usdt_subset, on='timestamp', how='inner')
        merged['close_raw'] = merged['close']
        merged['close'] = merged['close'] * merged['usdt_price']
        merged = merged.drop(columns=['usdt_price'])
        print(f"  Normalized {symbol_name}: {len(df)} -> {len(merged)} rows")
        return merged
    return df


def calculate_features(df):
    # deviation from $1 peg
    df['deviation_abs'] = (df['close'] - 1.0).abs()
    df['deviation_pct'] = (df['close'] - 1.0) / 1.0

    # volatility
    df['volatility_24h'] = df['close'].rolling(24).std()

    # time features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek

    # price momentum
    df['price_change_1h'] = df['close'].pct_change(1)
    df['price_change_6h'] = df['close'].pct_change(6)
    df['price_change_24h'] = df['close'].pct_change(24)

    # volume stats
    df['volume_ma_24h'] = df['volume'].rolling(24).mean()
    df['volume_std_24h'] = df['volume'].rolling(24).std()

    # lag features for forecasting
    for lag in [1, 2, 3, 6, 12, 24]:
        df[f'deviation_lag_{lag}h'] = df['deviation_pct'].shift(lag)
        df[f'volatility_lag_{lag}h'] = df['volatility_24h'].shift(lag)
        df[f'volume_lag_{lag}h'] = df['volume'].shift(lag)

    # velocity and z-scores
    df['deviation_velocity'] = df['deviation_pct'].diff()
    df['volume_velocity'] = df['volume'].pct_change()
    df['deviation_zscore_24h'] = (df['deviation_pct'] - df['deviation_pct'].rolling(24).mean()) / df['deviation_pct'].rolling(24).std()
    df['volume_zscore_24h'] = (df['volume'] - df['volume_ma_24h']) / df['volume_std_24h']

    # prediction targets (1h and 6h ahead)
    df['target_close_1h'] = df['close'].shift(-1)
    df['target_deviation_pct_1h'] = (df['target_close_1h'] - 1.0) / 1.0
    df['target_close_6h'] = df['close'].shift(-6)
    df['target_deviation_pct_6h'] = (df['target_close_6h'] - 1.0) / 1.0

    # stability labels based on future deviation
    def get_label(dev):
        if pd.isna(dev):
            return None
        d = abs(dev)
        if d < 0.001:
            return 'High'      # very stable
        elif d < 0.01:
            return 'Medium'    # minor deviation
        return 'Low'           # depeg risk

    df['target_label_1h'] = df['target_deviation_pct_1h'].apply(get_label)
    return df


def add_market_context(df, btc_df, eth_df):
    # merge BTC/ETH data for market context
    btc = btc_df[['timestamp', 'close', 'volume']].rename(columns={'close': 'btc_close', 'volume': 'btc_volume'})
    eth = eth_df[['timestamp', 'close', 'volume']].rename(columns={'close': 'eth_close', 'volume': 'eth_volume'})

    df = pd.merge(df, btc, on='timestamp', how='left')
    df = pd.merge(df, eth, on='timestamp', how='left')

    # rolling correlations
    df['corr_btc_24h'] = df['close'].rolling(24).corr(df['btc_close'])
    df['corr_eth_24h'] = df['close'].rolling(24).corr(df['eth_close'])

    # market returns
    df['btc_return_24h'] = df['btc_close'].pct_change(24)
    df['eth_return_24h'] = df['eth_close'].pct_change(24)

    return df


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # load market context (BTC/ETH)
    btc_path = os.path.join(RAW_DIR, 'BTC_USD_coinbase.csv')
    eth_path = os.path.join(RAW_DIR, 'ETH_USD_coinbase.csv')
    if not os.path.exists(btc_path):
        btc_path = os.path.join(RAW_DIR, 'BTC_USD_kraken.csv')
    if not os.path.exists(eth_path):
        eth_path = os.path.join(RAW_DIR, 'ETH_USD_kraken.csv')

    btc_df = load_data(btc_path) if os.path.exists(btc_path) else None
    eth_df = load_data(eth_path) if os.path.exists(eth_path) else None
    usdt_ref = get_usdt_reference(RAW_DIR)

    for filepath in glob.glob(os.path.join(RAW_DIR, '*.csv')):
        filename = os.path.basename(filepath)
        if 'BTC' in filename or 'ETH' in filename:
            continue

        print(f"Processing {filename}...")
        df = load_data(filepath)

        if usdt_ref is not None:
            df = normalize_to_usd(df, filename, usdt_ref)

        df = calculate_features(df)

        if btc_df is not None and eth_df is not None:
            df = add_market_context(df, btc_df, eth_df)

        df = df.dropna()
        save_path = os.path.join(PROCESSED_DIR, f"processed_{filename}")
        df.to_csv(save_path, index=False)
        print(f"Saved {save_path}")


if __name__ == "__main__":
    main()
