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
    'kraken': [
        'USDT/USD', 'USDC/USD', 'DAI/USD', 
        'BTC/USD', 'ETH/USD' # For correlation
    ],
    # Some stablecoins might not have direct USD pairs on Kraken or are more active on Binance
    # We will fetch what we can from Kraken, and others from Binance (against USDT usually, which is a limitation but acceptable)
    'binance': [
        'BUSD/USDT', 'TUSD/USDT', 'FRAX/USDT', 'USDP/USDT', 'GUSD/USDT'
    ]
}

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
        
    for exchange_name, symbols in TARGETS.items():
        for symbol in symbols:
            df = fetch_ohlcv(exchange_name, symbol, TIMEFRAME, SINCE_TIMESTAMP)
            if df is not None:
                # Clean filename: USDT/USD -> USDT_USD
                filename = f"{symbol.replace('/', '_')}_{exchange_name}.csv"
                filepath = os.path.join(DATA_DIR, filename)
                df.to_csv(filepath, index=False)
                print(f"Saved {symbol} to {filepath}")

if __name__ == "__main__":
    main()
