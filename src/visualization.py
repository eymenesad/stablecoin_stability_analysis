import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import confusion_matrix

RESULTS_FILE = 'reports/metrics/modeling_results_predictive.csv'
FIGURES_DIR = 'reports/figures'


def load_results():
    if not os.path.exists(RESULTS_FILE):
        print(f"Error: {RESULTS_FILE} not found. Run modeling.py first.")
        return None
    return pd.read_csv(RESULTS_FILE)


def plot_clustering(df):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='volatility_24h', y='deviation_abs',
                    hue='cluster', palette='viridis', alpha=0.6)
    plt.title('Clustering Results: Volatility vs Deviation')
    plt.xlabel('Volatility (24h)')
    plt.ylabel('Absolute Deviation')
    plt.yscale('log')
    plt.xscale('log')
    plt.savefig(os.path.join(FIGURES_DIR, 'clustering_clusters.png'))
    plt.close()
    print("Saved clustering_clusters.png")


def plot_classification_confusion(df):
    test_df = df.dropna(subset=['pred_risk', 'risk_label_1h'])

    if test_df.empty:
        print("No predictions found for classification.")
        return

    cm = confusion_matrix(test_df['risk_label_1h'], test_df['pred_risk'])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not-Low', 'Low'],
                yticklabels=['Not-Low', 'Low'])
    plt.title('Confusion Matrix (Binary Risk)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'classification_confusion_matrix.png'))
    plt.close()
    print("Saved classification_confusion_matrix.png")


def plot_anomaly_detection(df):
    for symbol in df['symbol'].unique():
        subset = df[df['symbol'] == symbol].copy()
        if subset.empty:
            continue

        subset['timestamp'] = pd.to_datetime(subset['timestamp'])
        subset = subset.sort_values('timestamp')

        plt.figure(figsize=(12, 6))
        plt.plot(subset['timestamp'], subset['deviation_pct'],
                 color='blue', alpha=0.5, label='Deviation %')

        anomalies = subset[subset['anomaly'] == -1]
        if not anomalies.empty:
            plt.scatter(anomalies['timestamp'], anomalies['deviation_pct'],
                       color='red', s=20, label='Anomaly', zorder=5)

        plt.title(f'Anomaly Detection: {symbol}')
        plt.xlabel('Date')
        plt.ylabel('Deviation %')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f'anomaly_detection_{symbol}.png'))
        plt.close()
        print(f"Saved anomaly_detection_{symbol}.png")


def plot_regression_comparison(df):
    test_df = df.dropna(subset=['pred_regression', 'pred_nn', 'target_deviation_pct_1h'])

    if test_df.empty:
        print("No predictions found for regression.")
        return

    # focus on meaningful deviations
    focus_df = test_df[test_df['target_deviation_pct_1h'].abs() > 0.001]
    if focus_df.empty:
        focus_df = test_df

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Ridge
    ax1.scatter(focus_df['target_deviation_pct_1h'], focus_df['pred_regression'],
                alpha=0.4, color='green')
    lims = [min(focus_df['target_deviation_pct_1h'].min(), focus_df['pred_regression'].min()),
            max(focus_df['target_deviation_pct_1h'].max(), focus_df['pred_regression'].max())]
    ax1.plot(lims, lims, 'k--', lw=2)
    ax1.set_title('Ridge: Actual vs Predicted (1h)')
    ax1.set_xlabel('Actual Deviation')
    ax1.set_ylabel('Predicted Deviation')
    ax1.grid(True, alpha=0.3)

    # NN
    ax2.scatter(focus_df['target_deviation_pct_1h'], focus_df['pred_nn'],
                alpha=0.4, color='purple')
    lims_nn = [min(focus_df['target_deviation_pct_1h'].min(), focus_df['pred_nn'].min()),
               max(focus_df['target_deviation_pct_1h'].max(), focus_df['pred_nn'].max())]
    ax2.plot(lims_nn, lims_nn, 'k--', lw=2)
    ax2.set_title('Neural Net: Actual vs Predicted (1h)')
    ax2.set_xlabel('Actual Deviation')
    ax2.set_ylabel('Predicted Deviation')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'regression_vs_nn_scatter.png'))
    plt.close()
    print("Saved regression_vs_nn_scatter.png")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    df = load_results()
    if df is not None:
        print("Generating visualizations...")
        plot_clustering(df)
        plot_classification_confusion(df)
        plot_anomaly_detection(df)
        plot_regression_comparison(df)
        print("Visualization complete.")


if __name__ == "__main__":
    main()
