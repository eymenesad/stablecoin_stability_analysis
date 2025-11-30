import requests
import ccxt
import pandas as pd
import time
from datetime import datetime, timedelta
import os

# Configuration
DATA_DIR = 'data/raw'
TIMEFRAME = '1h'
# Fetch last 730 days (approx 2 years)
DAYS_TO_FETCH = 730 
START_DATE = datetime.now() - timedelta(days=DAYS_TO_FETCH)
SINCE_TIMESTAMP = int(START_DATE.timestamp() * 1000)

# Target pairs. 
# We prefer USD pairs for stablecoins to measure real deviation.
# Kraken is good for USD pairs. Binance is good for liquid USDT pairs.
TARGETS = {
    'coinbase': [
        'USDT/USD', 'USDC/USD', 'DAI/USD', 
        'BTC/USD', 'ETH/USD' # For correlation
    ],
    'binance': [
        'BUSD/USDT', 'TUSD/USDT', 'USDP/USDT', 'USDE/USDT', 'FDUSD/USDT'
    ]
}

# CoinGecko Targets (ID -> Symbol Name for saving)
# We use 90 days to get hourly data (public API limitation)
COINGECKO_TARGETS = {
    'frax': 'FRAX_USD',
    'gemini-dollar': 'GUSD_USD'
}

def fetch_coingecko(coin_id, symbol_name):
    """
    Fetches OHLCV from CoinGecko.
    """
    print(f"Fetching {coin_id} ({symbol_name}) from CoinGecko...")
    # 1-90 days = hourly data. We use 30 to be safe and ensure hourly.
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc?vs_currency=usd&days=30"
    try:
        r = requests.get(url)
        if r.status_code != 200:
            print(f"  Error {r.status_code}: {r.text}")
            return None
            
        data = r.json()
        if not data:
            print(f"  No data found for {coin_id}")
            return None
            
        print(f"  Fetched {len(data)} candles.")
        
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])
        
        # Attempt to fetch volume
        print(f"  Fetching volume for {coin_id}...")
        v_url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days=30"
        vr = requests.get(v_url)
        if vr.status_code == 200:
            v_data = vr.json()
            v_df = pd.DataFrame(v_data['total_volumes'], columns=['timestamp', 'volume'])
            
            # Merge on timestamp
            df = pd.merge_asof(df.sort_values('timestamp'), v_df.sort_values('timestamp'), on='timestamp', direction='nearest', tolerance=3600000)
        else:
            print(f"  Volume fetch failed: {vr.status_code}")
            df['volume'] = 0
            
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
        
    except Exception as e:
        print(f"  Error fetching {coin_id}: {e}")
        return None

def fetch_ohlcv(exchange_id, symbol, timeframe, since):
    """
    Fetches historical OHLCV data handling pagination.
    """
    print(f"Fetching {symbol} from {exchange_id}...")
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class()
    
    all_ohlcv = []
    current_since = since
    
    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=1000)
            if not ohlcv:
                break
            
            all_ohlcv.extend(ohlcv)
            print(f"  Fetched {len(ohlcv)} candles. Last date: {datetime.fromtimestamp(ohlcv[-1][0]/1000)}")
            
            # Update since to the last timestamp + 1 timeframe duration
            # 1h = 3600 * 1000 ms
            last_timestamp = ohlcv[-1][0]
            current_since = last_timestamp + 3600000 
            
            # Break if we reached current time (allow some buffer)
            if last_timestamp >= (datetime.now().timestamp() * 1000) - 3600000:
                break
                
            # Rate limit
            time.sleep(exchange.rateLimit / 1000)
            
        except Exception as e:
            print(f"  Error fetching {symbol}: {e}")
            break
            
    if not all_ohlcv:
        print(f"No data found for {symbol}")
        return None

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def main():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    # CCXT Targets
    for exchange_name, symbols in TARGETS.items():
        for symbol in symbols:
            df = fetch_ohlcv(exchange_name, symbol, TIMEFRAME, SINCE_TIMESTAMP)
            if df is not None:
                # Clean filename: USDT/USD -> USDT_USD
                filename = f"{symbol.replace('/', '_')}_{exchange_name}.csv"
                filepath = os.path.join(DATA_DIR, filename)
                df.to_csv(filepath, index=False)
                print(f"Saved {symbol} to {filepath}")
                
    # CoinGecko Targets
    for coin_id, symbol_name in COINGECKO_TARGETS.items():
        df = fetch_coingecko(coin_id, symbol_name)
        if df is not None:
            filename = f"{symbol_name}_coingecko.csv"
            filepath = os.path.join(DATA_DIR, filename)
            df.to_csv(filepath, index=False)
            print(f"Saved {symbol_name} to {filepath}")

if __name__ == "__main__":
    main()
