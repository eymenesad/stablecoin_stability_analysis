import ccxt
import sys

def check(exchange_id, symbols):
    print(f"Checking {exchange_id}...")
    try:
        ex = getattr(ccxt, exchange_id)()
        ex.load_markets()
        for s in symbols:
            if s in ex.markets:
                print(f"  [OK] {s}")
            else:
                print(f"  [MISSING] {s}")
    except Exception as e:
        print(f"  Error: {e}")

check('coinbase', ['FRAX/USD', 'GUSD/USD'])
check('binance', ['USDe/USDT'])
