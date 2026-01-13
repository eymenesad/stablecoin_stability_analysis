import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob

PROCESSED_DIR = 'data/processed'
FIGURES_DIR = 'reports/figures'


def load_all_data():
    dfs = []
    for filepath in glob.glob(os.path.join(PROCESSED_DIR, '*.csv')):
        df = pd.read_csv(filepath)
        symbol = os.path.basename(filepath).replace('processed_', '').split('_')[0]
        df['symbol'] = symbol
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    combined['timestamp'] = pd.to_datetime(combined['timestamp'])
    return combined


def plot_price_history(df):
    plt.figure(figsize=(15, 8))
    sns.lineplot(data=df, x='timestamp', y='close', hue='symbol', alpha=0.7)
    plt.title('Stablecoin Price History (Last 2 Years)')
    plt.ylabel('Price (USD)')
    plt.xlabel('Date')
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'price_history.png'))
    print("Saved price_history.png")


def plot_depeg_events(df):
    depeg_df = df[df['deviation_pct'].abs() > 0.01]  # >1% deviation

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
    pivot = df.pivot_table(index='timestamp', columns='symbol', values='close')
    corr = pivot.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Price Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'correlation_heatmap.png'))
    print("Saved correlation_heatmap.png")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("Loading data...")
    df = load_all_data()

    print("Generating plots...")
    plot_price_history(df)
    plot_depeg_events(df)
    plot_correlation_heatmap(df)
    print("EDA complete.")


if __name__ == "__main__":
    main()
