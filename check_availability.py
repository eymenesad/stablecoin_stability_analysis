import ccxt
import sys

def check_binance():
    print("Checking Binance for USDe...")
    try:
        binance = ccxt.binance()
        binance.load_markets()
        # Check for USDe/USDT (case insensitive search)
        found = [s for s in binance.markets if 'USDE' in s and 'USDT' in s]
        print(f"  Found on Binance: {found}")
    except Exception as e:
        print(f"  Binance Error: {e}")

def check_coingecko():
    print("\nChecking CoinGecko via ccxt...")
    try:
        cg = ccxt.coingecko()
        # CoinGecko doesn't have load_markets in the same way, it has fetch_markets
        # But fetch_ohlcv usually takes a coin id and vs_currency
        # Let's try to fetch a sample candle for frax and gusd
        # Symbols in ccxt coingecko are usually 'base/quote' e.g. 'frax/usd'
        
        # We can't easily list all, but let's try to fetch OHLCV directly
        print("  Fetching FRAX/USD OHLCV...")
        ohlcv = cg.fetch_ohlcv('frax/usd', '1d', limit=5)
        print(f"  FRAX/USD: {len(ohlcv)} candles")
        
        print("  Fetching GUSD/USD OHLCV...")
        ohlcv = cg.fetch_ohlcv('gusd/usd', '1d', limit=5)
        print(f"  GUSD/USD: {len(ohlcv)} candles")
        
    except Exception as e:
        print(f"  CoinGecko Error: {e}")

if __name__ == "__main__":
    check_binance()
    check_coingecko()
