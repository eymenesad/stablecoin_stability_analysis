import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

PROCESSED_DIR = 'data/processed'
FIGURES_DIR = 'reports/figures'

def load_all_data():
    all_files = glob.glob(os.path.join(PROCESSED_DIR, '*.csv'))
    combined_df = pd.DataFrame()
    
    for filename in all_files:
        df = pd.read_csv(filename)
        # Extract symbol name from filename (e.g., processed_USDT_USD_kraken.csv -> USDT)
        base_name = os.path.basename(filename)
        symbol = base_name.replace('processed_', '').split('_')[0]
        df['symbol'] = symbol
        combined_df = pd.concat([combined_df, df], ignore_index=True)
        
    combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
    return combined_df

def plot_price_history(df):
    plt.figure(figsize=(15, 8))
    sns.lineplot(data=df, x='timestamp', y='close', hue='symbol', alpha=0.7)
    plt.title('Stablecoin Price History (Last 2 Years)')
    plt.ylabel('Price (USD/USDT)')
    plt.xlabel('Date')
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'price_history.png'))
    print("Saved price_history.png")

def plot_depeg_events(df):
    # Filter for significant deviation (> 1%)
    depeg_df = df[df['deviation_pct'].abs() > 0.01]
    
    plt.figure(figsize=(15, 8))
    sns.scatterplot(data=depeg_df, x='timestamp', y='close', hue='symbol', s=50)
    plt.title('Depeg Events (Deviation > 1%)')
    plt.ylabel('Price')
    plt.xlabel('Date')
    plt.axhline(y=1.0, color='r', linestyle='--')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'depeg_events.png'))
    print("Saved depeg_events.png")

def plot_correlation_heatmap(df):
    # Pivot table to get close prices for each symbol
    pivot_df = df.pivot_table(index='timestamp', columns='symbol', values='close')
    
    # Calculate correlation
    corr = pivot_df.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Price Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'correlation_heatmap.png'))
    print("Saved correlation_heatmap.png")

def main():
    if not os.path.exists(FIGURES_DIR):
        os.makedirs(FIGURES_DIR)
        
    print("Loading data...")
    df = load_all_data()
    
    print("Generating plots...")
    plot_price_history(df)
    plot_depeg_events(df)
    plot_correlation_heatmap(df)
    print("EDA Complete.")

if __name__ == "__main__":
    main()
