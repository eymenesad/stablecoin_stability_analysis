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
    Plots confusion matrix for Random Forest classification (Test Set only).
    """
    # Filter for rows where we have predictions (Test Set)
    test_df = df.dropna(subset=['pred_class'])
    
    if test_df.empty:
        print("No predictions found for classification.")
        return

    y_true = test_df['target_label_1h']
    y_pred = test_df['pred_class']
    
    labels = ['High', 'Medium', 'Low']
    # Check if all labels exist
    present_labels = [l for l in labels if l in y_true.unique() or l in y_pred.unique()]
    
    cm = confusion_matrix(y_true, y_pred, labels=present_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=present_labels)
    
    plt.figure(figsize=(8, 6))
    disp.plot(cmap='Blues')
    plt.title('Classification Confusion Matrix (Test Set - Future Prediction)')
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
    Compares Ridge vs NN predictions against Actual target (Test Set).
    """
    test_df = df.dropna(subset=['pred_regression', 'pred_nn'])
    
    if test_df.empty:
        print("No predictions found for regression.")
        return
    
    # Scatter plot: Actual vs Predicted
    plt.figure(figsize=(12, 5))
    
    # Ridge
    plt.subplot(1, 2, 1)
    plt.scatter(test_df['target_deviation_pct_1h'], test_df['pred_regression'], alpha=0.3, color='green')
    
    min_val = min(test_df['target_deviation_pct_1h'].min(), test_df['pred_regression'].min())
    max_val = max(test_df['target_deviation_pct_1h'].max(), test_df['pred_regression'].max())
    
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
    plt.title('Ridge Regression: Actual vs Predicted (1h Ahead)')
    plt.xlabel('Actual Deviation (1h ahead)')
    plt.ylabel('Predicted Deviation')
    
    # NN
    plt.subplot(1, 2, 2)
    plt.scatter(test_df['target_deviation_pct_1h'], test_df['pred_nn'], alpha=0.3, color='purple')
    
    min_val_nn = min(test_df['target_deviation_pct_1h'].min(), test_df['pred_nn'].min())
    max_val_nn = max(test_df['target_deviation_pct_1h'].max(), test_df['pred_nn'].max())
    
    plt.plot([min_val_nn, max_val_nn], [min_val_nn, max_val_nn], 'k--', lw=2)
    plt.title('Neural Network: Actual vs Predicted (1h Ahead)')
    plt.xlabel('Actual Deviation (1h ahead)')
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
