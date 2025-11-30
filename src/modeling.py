import pandas as pd
import numpy as np
import glob
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import classification_report, mean_squared_error, r2_score, silhouette_score, precision_score, recall_score, confusion_matrix
import joblib

PROCESSED_DIR = 'data/processed'
MODELS_DIR = 'models'
RESULTS_DIR = 'reports/metrics'

def load_and_prepare_data():
    all_files = glob.glob(os.path.join(PROCESSED_DIR, '*.csv'))
    combined_df = pd.DataFrame()
    
    for filename in all_files:
        df = pd.read_csv(filename)
        df = df.dropna()
        base_name = os.path.basename(filename)
        symbol = base_name.replace('processed_', '').replace('.csv', '')
        df['symbol'] = symbol
        
        # Filter small datasets
        if len(df) < 1000:
            print(f"Skipping {symbol}: too few rows ({len(df)})")
            continue
            
        combined_df = pd.concat([combined_df, df], ignore_index=True)
        
    combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
    combined_df = combined_df.sort_values('timestamp')
    return combined_df

def per_symbol_time_split(df, test_size=0.2):
    """
    Splits data chronologically PER SYMBOL.
    """
    train_dfs = []
    test_dfs = []
    
    for symbol in df['symbol'].unique():
        subset = df[df['symbol'] == symbol].sort_values('timestamp')
        split_idx = int(len(subset) * (1 - test_size))
        
        train_dfs.append(subset.iloc[:split_idx])
        test_dfs.append(subset.iloc[split_idx:])
        
    train_df = pd.concat(train_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)
    
    return train_df, test_df

def train_clustering(X):
    print("\n--- Clustering (K-Means) ---")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    score = silhouette_score(X, clusters)
    print(f"Silhouette Score: {score:.4f}")
    return kmeans, clusters

def train_classification(X_train, y_train, X_test, y_test):
    print("\n--- Classification (Random Forest) ---")
    rf = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=10,
                                  class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    return rf, y_pred

def evaluate_classification_detailed(y_true, y_pred):
    print("Classification Report (Test Set):")
    print(classification_report(y_true, y_pred))
    
    # Check for Low class specifically
    if 'Low' in y_true.values:
        low_mask = y_true == 'Low'
        low_acc = (y_pred[low_mask] == 'Low').mean()
        print(f"Recall on 'Low' class: {low_acc:.4f}")
    else:
        print("Warning: No 'Low' class samples in Test Set!")

def train_anomaly_detection(X, contamination=0.02):
    print("\n--- Anomaly Detection (Isolation Forest) ---")
    iso_forest = IsolationForest(contamination=contamination, random_state=42, n_jobs=-1)
    anomalies = iso_forest.fit_predict(X)
    return iso_forest, anomalies

def evaluate_anomaly_detection(df):
    """
    Evaluates anomaly detection against a 'Ground Truth' definition.
    Ground Truth: |deviation_pct| > 1.0% (0.01)
    """
    print("\n--- Anomaly Detection Evaluation ---")
    # Define Ground Truth
    df['is_depeg'] = df['deviation_pct'].abs() > 0.01
    
    # Anomaly output is -1 for anomaly, 1 for normal. Convert to boolean.
    df['pred_anomaly'] = df['anomaly'] == -1
    
    if df['is_depeg'].sum() == 0:
        print("No true depeg events (>1%) found in dataset.")
        return
        
    precision = precision_score(df['is_depeg'], df['pred_anomaly'])
    recall = recall_score(df['is_depeg'], df['pred_anomaly'])
    
    print(f"Ground Truth Depegs (>1%): {df['is_depeg'].sum()}")
    print(f"Detected Anomalies: {df['pred_anomaly'].sum()}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

def train_regression(X_train, y_train, X_test, y_test):
    print("\n--- Regression (Ridge) ---")
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    y_pred = ridge.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"Global RMSE: {rmse:.6f}, R2: {r2:.4f}")
    return ridge, y_pred

def train_neural_network(X_train, y_train, X_test, y_test):
    print("\n--- Neural Network (MLPRegressor) ---")
    mlp = MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu', solver='adam', 
                       alpha=0.001, max_iter=500, early_stopping=True, random_state=42)
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"Global RMSE: {rmse:.6f}, R2: {r2:.4f}")
    return mlp, y_pred

def evaluate_per_symbol(df, y_true_col, y_pred_col, task_type='classification'):
    print(f"\n--- Per-Symbol Evaluation ({task_type}) ---")
    symbols = df['symbol'].unique()
    results = []
    
    for sym in symbols:
        subset = df[df['symbol'] == sym]
        if subset.empty:
            continue
            
        y_true = subset[y_true_col]
        y_pred = subset[y_pred_col]
        
        if task_type == 'classification':
            acc = (y_true == y_pred).mean()
            print(f"{sym}: Accuracy = {acc:.4f}")
            results.append({'symbol': sym, 'accuracy': acc})
        elif task_type == 'regression':
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            print(f"{sym}: RMSE = {rmse:.6f}, R2 = {r2:.4f}")
            results.append({'symbol': sym, 'rmse': rmse, 'r2': r2})
            
    return pd.DataFrame(results)

def main():
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    print("Loading data...")
    df = load_and_prepare_data()
    
    if df.empty:
        print("No data found!")
        return

    # Feature Selection
    features = ['volatility_24h', 'price_change_1h', 'price_change_6h', 'price_change_24h',
                'volume_ma_24h', 'volume_std_24h', 'deviation_abs']
    
    # Add Lag Features
    for lag in [1, 2, 3, 6, 12, 24]:
        features.extend([f'deviation_lag_{lag}h', f'volatility_lag_{lag}h', f'volume_lag_{lag}h'])
    
    if 'btc_close' in df.columns:
        features.extend(['btc_close', 'eth_close', 'corr_btc_24h', 'corr_eth_24h'])
        
    print(f"Features: {len(features)} total features")
    
    # Ensure no NaNs or Infs in the selected features
    # This is critical because some rolling/lag operations might leave artifacts
    # or market data merging might have gaps
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=features)
    
    if df.empty:
        print("No data left after dropping NaNs!")
        return

    # 1. Clustering (Unsupervised)
    X_all = df[features]
    scaler = StandardScaler()
    X_all_scaled = scaler.fit_transform(X_all)
    
    kmeans, clusters = train_clustering(X_all_scaled)
    df['cluster'] = clusters
    
    # 2. Anomaly Detection (Unsupervised)
    # We evaluate this on the WHOLE dataset because it's unsupervised
    iso_model, anomalies = train_anomaly_detection(X_all_scaled, contamination=0.02)
    df['anomaly'] = anomalies
    evaluate_anomaly_detection(df)
    
    # Split Data for Supervised Learning (Per-Symbol)
    train_df, test_df = per_symbol_time_split(df)
    
    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
    
    X_train = scaler.transform(train_df[features])
    X_test = scaler.transform(test_df[features])
    
    # 3. Classification
    print("Training Classification Model...")
    y_train_class = train_df['target_label_1h']
    y_test_class = test_df['target_label_1h']
    
    rf_model, y_pred_class = train_classification(X_train, y_train_class, X_test, y_test_class)
    evaluate_classification_detailed(y_test_class, y_pred_class)
    
    test_df['pred_class'] = y_pred_class
    evaluate_per_symbol(test_df, 'target_label_1h', 'pred_class', 'classification')
    
    # 4. Regression
    print("Training Regression Models...")
    y_train_reg = train_df['target_deviation_pct_1h']
    y_test_reg = test_df['target_deviation_pct_1h']
    
    ridge_model, y_pred_ridge = train_regression(X_train, y_train_reg, X_test, y_test_reg)
    nn_model, y_pred_nn = train_neural_network(X_train, y_train_reg, X_test, y_test_reg)
    
    test_df['pred_regression'] = y_pred_ridge
    test_df['pred_nn'] = y_pred_nn
    
    evaluate_per_symbol(test_df, 'target_deviation_pct_1h', 'pred_regression', 'regression')
    
    # Merge test predictions back
    df['pred_class'] = np.nan
    df['pred_regression'] = np.nan
    df['pred_nn'] = np.nan
    
    df.loc[test_df.index, 'pred_class'] = test_df['pred_class']
    df.loc[test_df.index, 'pred_regression'] = test_df['pred_regression']
    df.loc[test_df.index, 'pred_nn'] = test_df['pred_nn']
    
    output_path = os.path.join(RESULTS_DIR, 'modeling_results_predictive.csv')
    df.to_csv(output_path, index=False)
    print(f"Saved results to {output_path}")

if __name__ == "__main__":
    main()
