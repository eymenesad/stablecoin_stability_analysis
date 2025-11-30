import requests
import time

def check_cg(coin_id):
    print(f"Checking {coin_id} on CoinGecko...")
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc?vs_currency=usd&days=30"
    try:
        r = requests.get(url)
        if r.status_code == 200:
            data = r.json()
            print(f"  Success! Got {len(data)} candles.")
            print(f"  Sample: {data[0]}")
        else:
            print(f"  Failed: {r.status_code} - {r.text}")
    except Exception as e:
        print(f"  Error: {e}")
    time.sleep(1)

check_cg('frax')
check_cg('gemini-dollar')
