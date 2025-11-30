import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Configuration
RESULTS_FILE = 'reports/metrics/modeling_results_with_predictions.csv'
FIGURES_DIR = 'reports/figures'

def load_results():
    if not os.path.exists(RESULTS_FILE):
        print(f"Error: {RESULTS_FILE} not found. Run modeling.py first.")
        return None
    return pd.read_csv(RESULTS_FILE)

def plot_clustering(df):
    """
    Plots clusters. Since we don't have PCA components in the CSV, 
    we'll plot Volatility vs Deviation colored by Cluster.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='volatility_24h', y='deviation_abs', hue='cluster', palette='viridis', alpha=0.6)
    plt.title('Clustering Results: Volatility vs Deviation')
    plt.xlabel('Volatility (24h)')
    plt.ylabel('Absolute Deviation')
    plt.savefig(os.path.join(FIGURES_DIR, 'clustering_clusters.png'))
    plt.close()
    print("Saved clustering_clusters.png")

def plot_classification_confusion(df):
    """
    Plots confusion matrix for Random Forest classification.
    """
    # Filter out rows where we might not have predictions (though we predicted on all)
    # Ensure labels match
    y_true = df['stability_label']
    y_pred = df['pred_class']
    
    cm = confusion_matrix(y_true, y_pred, labels=['High', 'Medium', 'Low'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['High', 'Medium', 'Low'])
    
    plt.figure(figsize=(8, 6))
    disp.plot(cmap='Blues')
    plt.title('Classification Confusion Matrix')
    plt.savefig(os.path.join(FIGURES_DIR, 'classification_confusion_matrix.png'))
    plt.close()
    print("Saved classification_confusion_matrix.png")

def plot_anomaly_detection(df):
    """
    Plots price deviation over time with anomalies highlighted.
    """
    # Take a subset or a specific symbol for clarity, e.g., USDC
    symbol = 'USDC'
    subset = df[df['symbol'] == symbol].copy()
    
    if subset.empty:
        # Fallback to first symbol
        symbol = df['symbol'].iloc[0]
        subset = df[df['symbol'] == symbol].copy()
        
    subset['timestamp'] = pd.to_datetime(subset['timestamp'])
    subset = subset.sort_values('timestamp')
    
    plt.figure(figsize=(12, 6))
    plt.plot(subset['timestamp'], subset['deviation_pct'], label='Deviation %', color='blue', alpha=0.5)
    
    # Anomalies
    anomalies = subset[subset['anomaly'] == -1]
    plt.scatter(anomalies['timestamp'], anomalies['deviation_pct'], color='red', label='Anomaly', s=20, zorder=5)
    
    plt.title(f'Anomaly Detection: {symbol} Deviation')
    plt.xlabel('Date')
    plt.ylabel('Deviation %')
    plt.legend()
    plt.savefig(os.path.join(FIGURES_DIR, 'anomaly_detection_timeseries.png'))
    plt.close()
    print("Saved anomaly_detection_timeseries.png")

def plot_regression_comparison(df):
    """
    Compares Ridge vs NN predictions against Actual target.
    """
    # Drop NaNs (rows where we couldn't predict target)
    df_reg = df.dropna(subset=['target_deviation', 'pred_regression', 'pred_nn'])
    
    # Scatter plot: Actual vs Predicted
    plt.figure(figsize=(12, 5))
    
    # Ridge
    plt.subplot(1, 2, 1)
    plt.scatter(df_reg['target_deviation'], df_reg['pred_regression'], alpha=0.3, color='green')
    plt.plot([df_reg['target_deviation'].min(), df_reg['target_deviation'].max()], 
             [df_reg['target_deviation'].min(), df_reg['target_deviation'].max()], 'k--', lw=2)
    plt.title('Ridge Regression: Actual vs Predicted')
    plt.xlabel('Actual Deviation')
    plt.ylabel('Predicted Deviation')
    
    # NN
    plt.subplot(1, 2, 2)
    plt.scatter(df_reg['target_deviation'], df_reg['pred_nn'], alpha=0.3, color='purple')
    plt.plot([df_reg['target_deviation'].min(), df_reg['target_deviation'].max()], 
             [df_reg['target_deviation'].min(), df_reg['target_deviation'].max()], 'k--', lw=2)
    plt.title('Neural Network: Actual vs Predicted')
    plt.xlabel('Actual Deviation')
    plt.ylabel('Predicted Deviation')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'regression_vs_nn_scatter.png'))
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
