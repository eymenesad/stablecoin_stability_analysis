import pandas as pd
import numpy as np
import glob
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestClassifier, IsolationForest, GradientBoostingClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (classification_report, mean_squared_error, r2_score,
                             silhouette_score, precision_score, recall_score, f1_score)

PROCESSED_DIR = 'data/processed'
MODELS_DIR = 'models'
RESULTS_DIR = 'reports/metrics'


def load_and_prepare_data():
    combined = pd.DataFrame()

    for filepath in glob.glob(os.path.join(PROCESSED_DIR, '*.csv')):
        df = pd.read_csv(filepath).dropna()
        symbol = os.path.basename(filepath).replace('processed_', '').replace('.csv', '')
        df['symbol'] = symbol

        if len(df) < 600:
            print(f"Skipping {symbol}: too few rows ({len(df)})")
            continue

        combined = pd.concat([combined, df], ignore_index=True)

    combined['timestamp'] = pd.to_datetime(combined['timestamp'])
    combined = combined.sort_values('timestamp').reset_index(drop=True)
    combined['row_id'] = combined.index
    return combined


def per_symbol_event_split(df, test_size=0.2, min_low_test=20, low_thresh=0.01):
    # chronological split per symbol, ensuring test has depeg events
    train_dfs, test_dfs = [], []

    for symbol in df['symbol'].unique():
        subset = df[df['symbol'] == symbol].sort_values('timestamp')
        if subset.empty:
            continue

        low_mask = (subset['target_label_1h'] == 'Low') | (subset['deviation_pct'].abs() > low_thresh)
        low_rows = subset[low_mask]

        split_idx = int(len(subset) * (1 - test_size))
        base_train = subset.iloc[:split_idx]
        base_test = subset.iloc[split_idx:]

        # push recent depeg rows to test if needed
        needed = max(min_low_test, int(len(low_rows) * 0.3)) if not low_rows.empty else 0
        if needed > 0:
            recent_low = low_rows.tail(needed)
            base_train = base_train.drop(index=recent_low.index, errors='ignore')
            base_test = pd.concat([base_test, recent_low]).drop_duplicates().sort_values('timestamp')

        train_dfs.append(base_train)
        test_dfs.append(base_test)

    return pd.concat(train_dfs, ignore_index=True), pd.concat(test_dfs, ignore_index=True)


def train_clustering(X):
    print("\n--- Clustering (K-Means) ---")
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    score = silhouette_score(X, clusters)
    print(f"Silhouette Score: {score:.4f}")
    return kmeans, clusters


def train_clustering_dbscan(X):
    print("\n--- Clustering (DBSCAN) ---")
    dbscan = DBSCAN(eps=0.5, min_samples=5, n_jobs=-1)
    clusters = dbscan.fit_predict(X)
    n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
    n_noise = (clusters == -1).sum()
    print(f"Number of Clusters: {n_clusters}, Noise points: {n_noise}")

    if n_clusters > 1 and (clusters != -1).sum() > 1:
        mask = clusters != -1
        score = silhouette_score(X[mask], clusters[mask])
        print(f"Silhouette Score (excl. noise): {score:.4f}")

    return dbscan, clusters


def compare_clustering_results(X, kmeans_clusters, dbscan_clusters):
    print("\n--- Clustering Comparison (K-Means vs DBSCAN) ---")

    kmeans_score = silhouette_score(X, kmeans_clusters)
    kmeans_n = len(set(kmeans_clusters))

    dbscan_n = len(set(dbscan_clusters)) - (1 if -1 in dbscan_clusters else 0)
    dbscan_noise = (dbscan_clusters == -1).sum()

    print(f"K-Means:  {kmeans_n} clusters, Silhouette = {kmeans_score:.4f}")

    if dbscan_n > 1 and (dbscan_clusters != -1).sum() > 1:
        mask = dbscan_clusters != -1
        dbscan_score = silhouette_score(X[mask], dbscan_clusters[mask])
        print(f"DBSCAN:   {dbscan_n} clusters, Silhouette = {dbscan_score:.4f}, Noise = {dbscan_noise}")
    else:
        print(f"DBSCAN:   {dbscan_n} clusters, Noise = {dbscan_noise} (silhouette N/A)")


def compare_classification_results(y_test, rf_pred, gb_pred, target_names=None):
    print("\n--- Classification Comparison (Random Forest vs Gradient Boosting) ---")

    from sklearn.metrics import accuracy_score

    rf_acc = accuracy_score(y_test, rf_pred)
    rf_f1 = f1_score(y_test, rf_pred, average='weighted')
    gb_acc = accuracy_score(y_test, gb_pred)
    gb_f1 = f1_score(y_test, gb_pred, average='weighted')

    print(f"\nRandom Forest:")
    print(f"  Accuracy: {rf_acc:.4f}, F1: {rf_f1:.4f}")
    print(classification_report(y_test, rf_pred, target_names=target_names))

    print(f"\nGradient Boosting:")
    print(f"  Accuracy: {gb_acc:.4f}, F1: {gb_f1:.4f}")
    print(classification_report(y_test, gb_pred, target_names=target_names))


def evaluate_anomaly_detection(df):
    print("\n--- Anomaly Detection Evaluation ---")
    df['is_depeg'] = df['deviation_pct'].abs() > 0.01
    df['pred_anomaly'] = df['anomaly'] == -1

    if df['is_depeg'].sum() == 0:
        print("No true depeg events found.")
        return

    prec = precision_score(df['is_depeg'], df['pred_anomaly'])
    rec = recall_score(df['is_depeg'], df['pred_anomaly'])

    print(f"Ground Truth Depegs (>1%): {df['is_depeg'].sum()}")
    print(f"Detected Anomalies: {df['pred_anomaly'].sum()}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")


def compare_anomaly_detection_results(df, iso_anomalies, lof_anomalies):
    print("\n--- Anomaly Detection Comparison (Isolation Forest vs LOF) ---")

    is_depeg = df['deviation_pct'].abs() > 0.01
    iso_pred = iso_anomalies == -1
    lof_pred = lof_anomalies == -1

    if is_depeg.sum() == 0:
        print("No true depeg events found.")
        return

    iso_p = precision_score(is_depeg, iso_pred, zero_division=0)
    iso_r = recall_score(is_depeg, iso_pred, zero_division=0)
    iso_f = f1_score(is_depeg, iso_pred, zero_division=0)

    lof_p = precision_score(is_depeg, lof_pred, zero_division=0)
    lof_r = recall_score(is_depeg, lof_pred, zero_division=0)
    lof_f = f1_score(is_depeg, lof_pred, zero_division=0)

    print(f"\nGround Truth Depegs (>1%): {is_depeg.sum()}")
    print(f"\nIsolation Forest:")
    print(f"  Detected: {iso_pred.sum()}, Precision: {iso_p:.4f}, Recall: {iso_r:.4f}, F1: {iso_f:.4f}")
    print(f"\nLocal Outlier Factor (LOF):")
    print(f"  Detected: {lof_pred.sum()}, Precision: {lof_p:.4f}, Recall: {lof_r:.4f}, F1: {lof_f:.4f}")


def evaluate_per_symbol(df, y_true_col, y_pred_col, task_type='classification'):
    print(f"\n--- Per-Symbol Evaluation ({task_type}) ---")

    for sym in df['symbol'].unique():
        subset = df[df['symbol'] == sym]
        if subset.empty:
            continue

        y_true = subset[y_true_col]
        y_pred = subset[y_pred_col]

        if task_type == 'classification':
            acc = (y_true == y_pred).mean()
            print(f"{sym}: Accuracy = {acc:.4f}")
        else:
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            print(f"{sym}: RMSE = {rmse:.6f}, R2 = {r2:.4f}")


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading data...")
    df = load_and_prepare_data()

    if df.empty:
        print("No data found!")
        return

    # feature columns
    features = ['volatility_24h', 'price_change_1h', 'price_change_6h', 'price_change_24h',
                'volume_ma_24h', 'volume_std_24h', 'deviation_velocity', 'volume_velocity',
                'deviation_zscore_24h', 'volume_zscore_24h']

    for lag in [1, 2, 3, 6, 12, 24]:
        features.extend([f'deviation_lag_{lag}h', f'volatility_lag_{lag}h', f'volume_lag_{lag}h'])

    if 'btc_close' in df.columns:
        features.extend(['btc_close', 'eth_close', 'corr_btc_24h', 'corr_eth_24h',
                        'btc_return_24h', 'eth_return_24h'])

    print(f"Features: {len(features)} total features")

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=features)
    df = df.reset_index(drop=True)
    df['row_id'] = df.index

    if df.empty:
        print("No data left after dropping NaNs!")
        return

    df['risk_label_1h'] = (df['target_label_1h'] == 'Low').astype(int)

    # scale features
    X_all = df[features]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)

    # === CLUSTERING ===
    kmeans, kmeans_clusters = train_clustering(X_scaled)
    df['cluster'] = kmeans_clusters

    dbscan, dbscan_clusters = train_clustering_dbscan(X_scaled)
    df['cluster_dbscan'] = dbscan_clusters

    compare_clustering_results(X_scaled, kmeans_clusters, dbscan_clusters)

    # === ANOMALY DETECTION (per-symbol tuning) ===
    print("\n--- Tuning Anomaly Detection Per Symbol ---")
    df['anomaly'] = 1

    for symbol in df['symbol'].unique():
        idx = df[df['symbol'] == symbol].index
        X_sub = X_scaled[idx]
        is_depeg = df.loc[idx, 'deviation_pct'].abs() > 0.01

        if is_depeg.sum() == 0:
            iso = IsolationForest(contamination=0.005, random_state=42, n_jobs=-1)
            df.loc[idx, 'anomaly'] = iso.fit_predict(X_sub)
            print(f"{symbol}: No depegs. Used default contamination 0.005.")
            continue

        # sweep contamination to find best F1
        best_f1, best_model = -1, None
        for cont in [0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2]:
            iso = IsolationForest(contamination=cont, random_state=42, n_jobs=-1)
            preds = iso.fit_predict(X_sub) == -1
            if preds.sum() > 0:
                p = precision_score(is_depeg, preds, zero_division=0)
                r = recall_score(is_depeg, preds, zero_division=0)
                f = 2 * p * r / (p + r) if (p + r) > 0 else 0
                if f > best_f1:
                    best_f1, best_model = f, iso

        print(f"{symbol}: Best F1={best_f1:.4f}")
        df.loc[idx, 'anomaly'] = best_model.fit_predict(X_sub)

    evaluate_anomaly_detection(df)

    # LOF comparison
    print("\n--- Training LOF for Comparison ---")
    df['anomaly_lof'] = 1

    for symbol in df['symbol'].unique():
        idx = df[df['symbol'] == symbol].index
        X_sub = X_scaled[idx]
        is_depeg = df.loc[idx, 'deviation_pct'].abs() > 0.01

        if is_depeg.sum() == 0:
            lof = LocalOutlierFactor(n_neighbors=20, contamination=0.005, novelty=False, n_jobs=-1)
            df.loc[idx, 'anomaly_lof'] = lof.fit_predict(X_sub)
            continue

        best_f1, best_preds = -1, None
        for cont in [0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2]:
            lof = LocalOutlierFactor(n_neighbors=20, contamination=cont, novelty=False, n_jobs=-1)
            preds = lof.fit_predict(X_sub)
            pred_bool = preds == -1
            if pred_bool.sum() > 0:
                p = precision_score(is_depeg, pred_bool, zero_division=0)
                r = recall_score(is_depeg, pred_bool, zero_division=0)
                f = 2 * p * r / (p + r) if (p + r) > 0 else 0
                if f > best_f1:
                    best_f1, best_preds = f, preds

        print(f"{symbol}: LOF Best F1={best_f1:.4f}")
        df.loc[idx, 'anomaly_lof'] = best_preds

    compare_anomaly_detection_results(df, df['anomaly'].values, df['anomaly_lof'].values)

    # === TRAIN/TEST SPLIT ===
    train_df, test_df = per_symbol_event_split(df)
    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

    # oversample Low class
    print("Oversampling 'Low' class in training data...")
    low_train = train_df[train_df['target_label_1h'] == 'Low']
    if not low_train.empty:
        oversampled = pd.concat([low_train] * 10, ignore_index=True)
        train_df_balanced = pd.concat([train_df, oversampled], ignore_index=True)
        print(f"  Added {len(oversampled)} 'Low' samples. New train size: {len(train_df_balanced)}")
    else:
        train_df_balanced = train_df

    X_train = scaler.transform(train_df_balanced[features])
    X_test = scaler.transform(test_df[features])

    # === CLASSIFICATION ===
    print("\nTraining Classification Models (1h Horizon)...")
    y_train = train_df_balanced['risk_label_1h']
    y_test = test_df['risk_label_1h']
    threshold = 0.005

    # Random Forest
    print("\n--- Random Forest Classification ---")
    rf = RandomForestClassifier(n_estimators=500, max_depth=22, min_samples_split=3,
                                class_weight={0: 1, 1: 20}, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    proba_rf = rf.predict_proba(X_test)[:, 1]
    y_pred_rf = (proba_rf >= threshold).astype(int)

    print("Classification Report (Random Forest, Binary Risk, Test Set):")
    print(classification_report(y_test, y_pred_rf, target_names=['Not-Low', 'Low']))
    print(f"Threshold used for Low: {threshold}")
    print(f"Debug RF: Test Low count {(y_test == 1).sum()}")
    print(f"Debug RF: Pred positives {(y_pred_rf == 1).sum()}")
    print(f"Debug RF: True Low captured {((y_test == 1) & (y_pred_rf == 1)).sum()}")

    # Gradient Boosting
    print("\n--- Gradient Boosting Classification ---")
    sample_weights = np.where(y_train == 1, 20.0, 1.0)
    gb = GradientBoostingClassifier(n_estimators=300, max_depth=8, learning_rate=0.1,
                                    min_samples_split=5, random_state=42)
    gb.fit(X_train, y_train, sample_weight=sample_weights)
    proba_gb = gb.predict_proba(X_test)[:, 1]
    y_pred_gb = (proba_gb >= threshold).astype(int)

    print("Classification Report (Gradient Boosting, Binary Risk, Test Set):")
    print(classification_report(y_test, y_pred_gb, target_names=['Not-Low', 'Low']))
    print(f"Threshold used for Low: {threshold}")
    print(f"Debug GB: Test Low count {(y_test == 1).sum()}")
    print(f"Debug GB: Pred positives {(y_pred_gb == 1).sum()}")
    print(f"Debug GB: True Low captured {((y_test == 1) & (y_pred_gb == 1)).sum()}")

    compare_classification_results(y_test, y_pred_rf, y_pred_gb, target_names=['Not-Low', 'Low'])

    test_df['pred_risk'] = y_pred_rf
    test_df['pred_risk_gb'] = y_pred_gb
    evaluate_per_symbol(test_df, 'risk_label_1h', 'pred_risk', 'classification')

    # === REGRESSION ===
    print("\nTraining Regression Model (1h Horizon) with magnitude weighting...")
    reg_train = train_df.copy()
    reg_test = test_df.copy()
    y_train_reg = reg_train['target_deviation_pct_1h']
    y_test_reg = reg_test['target_deviation_pct_1h']

    # weight larger deviations more
    reg_weights = 1.0 + 50.0 * y_train_reg.abs()

    ridge = Ridge(alpha=0.5)
    ridge.fit(scaler.transform(reg_train[features]), y_train_reg, sample_weight=reg_weights)
    y_pred_ridge = ridge.predict(scaler.transform(reg_test[features]))

    rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_ridge))
    r2 = r2_score(y_test_reg, y_pred_ridge)
    print(f"Regression (1h) RMSE: {rmse:.6f}, R2: {r2:.4f}")

    reg_test['pred_regression'] = y_pred_ridge
    evaluate_per_symbol(reg_test, 'target_deviation_pct_1h', 'pred_regression', 'regression')

    # Neural Network
    print("\nTraining Neural Network Regression (1h Horizon)...")
    nn = MLPRegressor(hidden_layer_sizes=(128, 64), activation='relu', solver='adam',
                      alpha=0.001, max_iter=600, early_stopping=True, random_state=42)
    nn.fit(scaler.transform(reg_train[features]), y_train_reg)
    y_pred_nn = nn.predict(scaler.transform(reg_test[features]))

    reg_test['pred_nn'] = y_pred_nn
    rmse_nn = np.sqrt(mean_squared_error(y_test_reg, y_pred_nn))
    r2_nn = r2_score(y_test_reg, y_pred_nn)
    print(f"NN Regression (1h) RMSE: {rmse_nn:.6f}, R2: {r2_nn:.4f}")

    # merge predictions back to main df
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
