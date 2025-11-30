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
        
        # Filter very small datasets
        if len(df) < 600:
            print(f"Skipping {symbol}: too few rows ({len(df)})")
            continue
            
        combined_df = pd.concat([combined_df, df], ignore_index=True)
        
    combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
    combined_df = combined_df.sort_values('timestamp').reset_index(drop=True)
    combined_df['row_id'] = combined_df.index  # stable id for merges
    return combined_df

def per_symbol_event_split(df, test_size=0.2, min_low_test=20, low_thresh=0.01):
    """
    Chronological split per symbol, but ensure test contains depeg/low events.
    - Base split: last `test_size` fraction.
    - If Low events are scarce in base test, move the most recent Low rows into test.
    """
    train_dfs = []
    test_dfs = []

    for symbol in df['symbol'].unique():
        subset = df[df['symbol'] == symbol].sort_values('timestamp')
        if subset.empty:
            continue

        # Identify low/depeg rows
        low_mask = (subset['target_label_1h'] == 'Low') | (subset['deviation_pct'].abs() > low_thresh)
        low_rows = subset[low_mask]

        # Base chronological split
        split_idx = int(len(subset) * (1 - test_size))
        base_train = subset.iloc[:split_idx]
        base_test = subset.iloc[split_idx:]

        # If Low rows are missing or too few in test, push the most recent Low rows to test
        needed = max(min_low_test, int(len(low_rows) * 0.3)) if not low_rows.empty else 0
        if needed > 0:
            recent_low = low_rows.tail(needed)
            # Move these from train to test (avoid duplicates)
            base_train = base_train.drop(index=recent_low.index, errors='ignore')
            base_test = pd.concat([base_test, recent_low]).drop_duplicates().sort_values('timestamp')

        train_dfs.append(base_train)
        test_dfs.append(base_test)

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

    # Feature Selection (focus on velocities/lags; drop current deviation_abs to reduce leakage)
    features = ['volatility_24h', 'price_change_1h', 'price_change_6h', 'price_change_24h',
                'volume_ma_24h', 'volume_std_24h',
                'deviation_velocity', 'volume_velocity', 'deviation_zscore_24h', 'volume_zscore_24h']
    
    # Add Lag Features
    for lag in [1, 2, 3, 6, 12, 24]:
        features.extend([f'deviation_lag_{lag}h', f'volatility_lag_{lag}h', f'volume_lag_{lag}h'])
    
    if 'btc_close' in df.columns:
        features.extend(['btc_close', 'eth_close', 'corr_btc_24h', 'corr_eth_24h', 'btc_return_24h', 'eth_return_24h'])
        
    print(f"Features: {len(features)} total features")
    
    # Ensure no NaNs or Infs in the selected features
    # This is critical because some rolling/lag operations might leave artifacts
    # or market data merging might have gaps
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=features)
    df = df.reset_index(drop=True)
    df['row_id'] = df.index  # reset after dropna so ids match current rows
    
    if df.empty:
        print("No data left after dropping NaNs!")
        return

    # Binary risk label: Low vs Not-Low
    df['risk_label_1h'] = (df['target_label_1h'] == 'Low').astype(int)

    # 1. Clustering (Unsupervised)
    X_all = df[features]
    scaler = StandardScaler()
    X_all_scaled = scaler.fit_transform(X_all)
    
    kmeans, clusters = train_clustering(X_all_scaled)
    df['cluster'] = clusters
    
    # 2. Anomaly Detection (Unsupervised but Tuned)
    print("\n--- Tuning Anomaly Detection Per Symbol ---")
    
    df['anomaly'] = 1 # Default normal
    
    for symbol in df['symbol'].unique():
        subset_idx = df[df['symbol'] == symbol].index
        subset_X = X_all_scaled[subset_idx]
        subset_df = df.loc[subset_idx]
        
        # Ground truth: Deviation > 1%
        is_depeg = subset_df['deviation_pct'].abs() > 0.01
        
        if is_depeg.sum() == 0:
            iso = IsolationForest(contamination=0.005, random_state=42, n_jobs=-1)
            anoms = iso.fit_predict(subset_X)
            df.loc[subset_idx, 'anomaly'] = anoms
            print(f"{symbol}: No depegs. Used default contamination 0.005.")
            continue
            
        best_f1 = -1
        best_cont = 0.01
        best_model = None
        
        # Sweep contamination
        for cont in [0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2]:
            iso = IsolationForest(contamination=cont, random_state=42, n_jobs=-1)
            preds = iso.fit_predict(subset_X)
            pred_bool = preds == -1
            
            f1 = 0
            if pred_bool.sum() > 0:
                prec = precision_score(is_depeg, pred_bool, zero_division=0)
                rec = recall_score(is_depeg, pred_bool, zero_division=0)
                if prec + rec > 0:
                    f1 = 2 * (prec * rec) / (prec + rec)
            
            if f1 > best_f1:
                best_f1 = f1
                best_cont = cont
                best_model = iso
        
        print(f"{symbol}: Best Contamination={best_cont}, F1={best_f1:.4f}")
        final_preds = best_model.fit_predict(subset_X)
        df.loc[subset_idx, 'anomaly'] = final_preds

    evaluate_anomaly_detection(df)
    
    # Split Data for Supervised Learning (Per-Symbol Event Split)
    train_df, test_df = per_symbol_event_split(df)
    
    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
    
    # Oversample 'Low' class in Training Data
    print("Oversampling 'Low' class in training data...")
    low_train = train_df[train_df['target_label_1h'] == 'Low']
    if not low_train.empty:
        # Oversample by 10x or up to 20% of data? Let's just duplicate 10 times for now
        # A better approach is SMOTE, but we'll do simple duplication to avoid dependency issues
        oversampled_low = pd.concat([low_train] * 10, ignore_index=True)
        train_df_balanced = pd.concat([train_df, oversampled_low], ignore_index=True)
        print(f"  Added {len(oversampled_low)} 'Low' samples. New train size: {len(train_df_balanced)}")
    else:
        print("  No 'Low' samples in training data to oversample.")
        train_df_balanced = train_df

    X_train = scaler.transform(train_df_balanced[features])
    X_test = scaler.transform(test_df[features])
    
    # 3. Classification (1h Horizon) - Binary risk
    print("\nTraining Classification Model (1h Horizon)...")
    y_train_class = train_df_balanced['risk_label_1h']
    y_test_class = test_df['risk_label_1h']
    
    # Stronger weight for Low (risk=1)
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=22,
        min_samples_split=3,
        class_weight={0:1, 1:20},
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train_class)
    proba = rf.predict_proba(X_test)[:,1]
    # Aggressive threshold to favor recall on risk
    threshold = 0.005
    y_pred_class = (proba >= threshold).astype(int)
    
    print("Classification Report (Binary Risk, Test Set):")
    print(classification_report(y_test_class, y_pred_class, target_names=['Not-Low','Low']))
    print(f"Threshold used for Low: {threshold}")

    print("Debug: Test Low count", (y_test_class==1).sum())
    print("Debug: Pred positives", (y_pred_class==1).sum())
    print("Debug: True Low captured", ((y_test_class==1) & (y_pred_class==1)).sum())
    
    test_df['pred_risk'] = y_pred_class
    evaluate_per_symbol(test_df, 'risk_label_1h', 'pred_risk', 'classification')

    # 4. Regression (1h Horizon) with magnitude weighting to avoid mean collapse
    print("\nTraining Regression Model (1h Horizon) with magnitude weighting...")
    reg_train = train_df.copy()
    reg_test = test_df.copy()
    y_train_reg = reg_train['target_deviation_pct_1h']
    y_test_reg = reg_test['target_deviation_pct_1h']
    reg_weights = 1.0 + 50.0 * y_train_reg.abs()
    ridge_reg = Ridge(alpha=0.5)
    ridge_reg.fit(scaler.transform(reg_train[features]), y_train_reg, sample_weight=reg_weights)
    y_pred_reg = ridge_reg.predict(scaler.transform(reg_test[features]))
    rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
    r2 = r2_score(y_test_reg, y_pred_reg)
    print(f"Regression (1h) RMSE: {rmse:.6f}, R2: {r2:.4f}")
    reg_test['pred_regression'] = y_pred_reg
    evaluate_per_symbol(reg_test, 'target_deviation_pct_1h', 'pred_regression', 'regression')

    # 5. Neural Network Regression (1h Horizon) on same weighting
    print("\nTraining Neural Network Regression (1h Horizon)...")
    nn_model = MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu', solver='adam',
                            alpha=0.001, max_iter=600, early_stopping=True, random_state=42)
    nn_model.fit(scaler.transform(reg_train[features]), y_train_reg)  # sample_weight optional; omit for stability
    y_pred_nn = nn_model.predict(scaler.transform(reg_test[features]))
    reg_test['pred_nn'] = y_pred_nn
    rmse_nn = np.sqrt(mean_squared_error(y_test_reg, y_pred_nn))
    r2_nn = r2_score(y_test_reg, y_pred_nn)
    print(f"NN Regression (1h) RMSE: {rmse_nn:.6f}, R2: {r2_nn:.4f}")

    # Merge test predictions back
    df['pred_risk'] = np.nan
    df.loc[test_df['row_id'], 'pred_risk'] = test_df['pred_risk'].values
    df['pred_regression'] = np.nan
    df.loc[reg_test['row_id'], 'pred_regression'] = reg_test['pred_regression'].values
    df['pred_nn'] = np.nan
    df.loc[reg_test['row_id'], 'pred_nn'] = reg_test['pred_nn'].values
    
    output_path = os.path.join(RESULTS_DIR, 'modeling_results_predictive.csv')
    df.to_csv(output_path, index=False)
    print(f"Saved results to {output_path}")

if __name__ == "__main__":
    main()
