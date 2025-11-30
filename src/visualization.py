import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Configuration
RESULTS_FILE = 'reports/metrics/modeling_results_predictive.csv'
FIGURES_DIR = 'reports/figures'

def load_results():
    if not os.path.exists(RESULTS_FILE):
        print(f"Error: {RESULTS_FILE} not found. Run modeling.py first.")
        return None
    return pd.read_csv(RESULTS_FILE)

def plot_clustering(df):
    """
    Plots clusters.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='volatility_24h', y='deviation_abs', hue='cluster', palette='viridis', alpha=0.6)
    plt.title('Clustering Results: Volatility vs Deviation')
    plt.xlabel('Volatility (24h)')
    plt.ylabel('Absolute Deviation')
    plt.yscale('log') # Log scale might be better for deviation
    plt.xscale('log')
    plt.savefig(os.path.join(FIGURES_DIR, 'clustering_clusters.png'))
    plt.close()
    print("Saved clustering_clusters.png")

def plot_classification_confusion(df):
    """
    Plots confusion matrix for the Test Set (Binary Risk).
    """
    test_df = df.dropna(subset=['pred_risk', 'risk_label_1h'])
    
    if test_df.empty:
        print("No predictions found for classification.")
        return

    cm = confusion_matrix(test_df['risk_label_1h'], test_df['pred_risk'])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not-Low', 'Low'], 
                yticklabels=['Not-Low', 'Low'])
    plt.title('Confusion Matrix (Binary Risk Prediction)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'classification_confusion_matrix.png'))
    plt.close()
    print("Saved classification_confusion_matrix.png")

def plot_anomaly_detection(df):
    """
    Plots price deviation over time with anomalies highlighted.
    """
    # Plot for each symbol
    symbols = df['symbol'].unique()
    
    for symbol in symbols:
        subset = df[df['symbol'] == symbol].copy()
        if subset.empty:
            continue
            
        subset['timestamp'] = pd.to_datetime(subset['timestamp'])
        subset = subset.sort_values('timestamp')
        
        plt.figure(figsize=(12, 6))
        plt.plot(subset['timestamp'], subset['deviation_pct'], label='Deviation %', color='blue', alpha=0.5)
        
        # Anomalies
        anomalies = subset[subset['anomaly'] == -1]
        if not anomalies.empty:
            plt.scatter(anomalies['timestamp'], anomalies['deviation_pct'], color='red', label='Anomaly', s=20, zorder=5)
        
        plt.title(f'Anomaly Detection: {symbol} Deviation')
        plt.xlabel('Date')
        plt.ylabel('Deviation %')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, f'anomaly_detection_{symbol}.png'))
        plt.close()
        print(f"Saved anomaly_detection_{symbol}.png")

def plot_regression_comparison(df):
    """
    Compares Ridge and NN predictions against Actual target (Test Set).
    """
    test_df = df.dropna(subset=['pred_regression', 'pred_nn', 'target_deviation_pct_1h'])
    
    if test_df.empty:
        print("No predictions found for regression.")
        return

    # Focus on meaningful deviations to reduce the near-zero cross
    focus_df = test_df[test_df['target_deviation_pct_1h'].abs() > 0.001]
    if focus_df.empty:
        focus_df = test_df

    # Scatter plots: Ridge and NN
    plt.figure(figsize=(12, 5))
    
    # Ridge
    plt.subplot(1, 2, 1)
    plt.scatter(focus_df['target_deviation_pct_1h'], focus_df['pred_regression'], alpha=0.4, color='green', label='Ridge')
    
    min_val = min(focus_df['target_deviation_pct_1h'].min(), focus_df['pred_regression'].min())
    max_val = max(focus_df['target_deviation_pct_1h'].max(), focus_df['pred_regression'].max())
    
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Perfect Prediction')
    plt.title('Ridge Regression: Actual vs Predicted (1h Ahead)')
    plt.xlabel('Actual Deviation (1h)')
    plt.ylabel('Predicted Deviation (1h)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # NN
    plt.subplot(1, 2, 2)
    plt.scatter(focus_df['target_deviation_pct_1h'], focus_df['pred_nn'], alpha=0.4, color='purple', label='NN')
    
    min_val_nn = min(focus_df['target_deviation_pct_1h'].min(), focus_df['pred_nn'].min())
    max_val_nn = max(focus_df['target_deviation_pct_1h'].max(), focus_df['pred_nn'].max())
    
    plt.plot([min_val_nn, max_val_nn], [min_val_nn, max_val_nn], 'k--', lw=2, label='Perfect Prediction')
    plt.title('Neural Net: Actual vs Predicted (1h Ahead)')
    plt.xlabel('Actual Deviation (1h)')
    plt.ylabel('Predicted Deviation (1h)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'regression_vs_nn_scatter.png'))
    plt.close()
    
    # Plot 6h Forecast
    if 'pred_regression_6h' in df.columns:
        plt.figure(figsize=(10, 6))
        test_df_6h = df.dropna(subset=['pred_regression_6h'])
        test_df_6h_focus = test_df_6h[test_df_6h['target_deviation_pct_6h'].abs() > 0.001]
        if test_df_6h_focus.empty:
            test_df_6h_focus = test_df_6h
        plt.scatter(test_df_6h_focus['target_deviation_pct_6h'], test_df_6h_focus['pred_regression_6h'], alpha=0.4, color='purple', label='Ridge (6h)')
        
        min_val = min(test_df_6h_focus['target_deviation_pct_6h'].min(), test_df_6h_focus['pred_regression_6h'].min())
        max_val = max(test_df_6h_focus['target_deviation_pct_6h'].max(), test_df_6h_focus['pred_regression_6h'].max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        
        plt.title('Regression Forecast (6h Ahead) vs Actual Deviation')
        plt.xlabel('Actual Deviation (6h)')
        plt.ylabel('Predicted Deviation (6h)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(FIGURES_DIR, 'regression_6h_scatter.png'))
        plt.close()
    print("Saved regression_vs_nn_scatter.png")

def main():
    if not os.path.exists(FIGURES_DIR):
        os.makedirs(FIGURES_DIR)
        
    df = load_results()
    if df is not None:
        print("Generating visualizations...")
        plot_clustering(df)
        plot_classification_confusion(df)
        plot_anomaly_detection(df)
        plot_regression_comparison(df)
        print("Visualization Complete.")

if __name__ == "__main__":
    main()
