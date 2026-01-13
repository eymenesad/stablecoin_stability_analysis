import ccxt
import pandas as pd
import requests
import time
import os
from datetime import datetime, timedelta

DATA_DIR = 'data/raw'
TIMEFRAME = '1h'
DAYS_TO_FETCH = 730  # ~2 years
START_DATE = datetime.now() - timedelta(days=DAYS_TO_FETCH)
SINCE_TIMESTAMP = int(START_DATE.timestamp() * 1000)

TARGETS = {
    'coinbase': ['USDT/USD', 'USDC/USD', 'DAI/USD', 'BTC/USD', 'ETH/USD'],
    'binance': ['BUSD/USDT', 'TUSD/USDT', 'USDP/USDT', 'USDE/USDT', 'FDUSD/USDT']
}

COINGECKO_TARGETS = {
    'frax': 'FRAX_USD',
    'gemini-dollar': 'GUSD_USD'
}


def fetch_coingecko(coin_id, symbol_name):
    print(f"Fetching {coin_id} ({symbol_name}) from CoinGecko...")
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc?vs_currency=usd&days=30"
    
    try:
        r = requests.get(url)
        if r.status_code != 200:
            print(f"  Error {r.status_code}: {r.text}")
            return None

        data = r.json()
        if not data:
            print(f"  No data for {coin_id}")
            return None

        print(f"  Got {len(data)} candles")
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close'])

        # try to get volume separately
        v_url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days=30"
        vr = requests.get(v_url)
        if vr.status_code == 200:
            v_data = vr.json()
            v_df = pd.DataFrame(v_data['total_volumes'], columns=['timestamp', 'volume'])
            df = pd.merge_asof(df.sort_values('timestamp'), v_df.sort_values('timestamp'),
                              on='timestamp', direction='nearest', tolerance=3600000)
        else:
            df['volume'] = 0

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df

    except Exception as e:
        print(f"  Error: {e}")
        return None


def fetch_ohlcv(exchange_id, symbol, timeframe, since):
    print(f"Fetching {symbol} from {exchange_id}...")
    exchange = getattr(ccxt, exchange_id)()

    all_ohlcv = []
    current_since = since

    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=1000)
            if not ohlcv:
                break

            all_ohlcv.extend(ohlcv)
            last_ts = ohlcv[-1][0]
            print(f"  {len(ohlcv)} candles, last: {datetime.fromtimestamp(last_ts/1000)}")

            current_since = last_ts + 3600000  # next hour

            if last_ts >= (datetime.now().timestamp() * 1000) - 3600000:
                break

            time.sleep(exchange.rateLimit / 1000)

        except Exception as e:
            print(f"  Error: {e}")
            break

    if not all_ohlcv:
        print(f"No data for {symbol}")
        return None

    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    for exchange_name, symbols in TARGETS.items():
        for symbol in symbols:
            df = fetch_ohlcv(exchange_name, symbol, TIMEFRAME, SINCE_TIMESTAMP)
            if df is not None:
                filename = f"{symbol.replace('/', '_')}_{exchange_name}.csv"
                df.to_csv(os.path.join(DATA_DIR, filename), index=False)
                print(f"Saved {filename}")

    for coin_id, symbol_name in COINGECKO_TARGETS.items():
        df = fetch_coingecko(coin_id, symbol_name)
        if df is not None:
            filename = f"{symbol_name}_coingecko.csv"
            df.to_csv(os.path.join(DATA_DIR, filename), index=False)
            print(f"Saved {filename}")


if __name__ == "__main__":
    main()
