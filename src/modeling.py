import pandas as pd
import numpy as np
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import Ridge
from sklearn.metrics import classification_report, mean_squared_error, r2_score, silhouette_score
import joblib

PROCESSED_DIR = 'data/processed'
MODELS_DIR = 'models'
RESULTS_DIR = 'reports/metrics'

def load_and_prepare_data():
    all_files = glob.glob(os.path.join(PROCESSED_DIR, '*.csv'))
    combined_df = pd.DataFrame()
    
    for filename in all_files:
        df = pd.read_csv(filename)
        # Drop rows with NaN (often first 24 rows due to rolling window)
        df = df.dropna()
        base_name = os.path.basename(filename)
        symbol = base_name.replace('processed_', '').split('_')[0]
        df['symbol'] = symbol
        combined_df = pd.concat([combined_df, df], ignore_index=True)
        
    return combined_df

def train_clustering(X):
    print("\n--- Clustering (K-Means) ---")
    # Determine optimal K (simplified: using K=3 for High/Med/Low stability logic)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)
    
    score = silhouette_score(X, clusters)
    print(f"Silhouette Score: {score:.4f}")
    return kmeans, clusters

def train_classification(X, y):
    print("\n--- Classification (Random Forest) ---")
    # Time-series split (shuffle=False)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    return rf

def train_anomaly_detection(X):
    print("\n--- Anomaly Detection (Isolation Forest) ---")
    # Contamination: expected proportion of outliers. We assume ~1% are depegs.
    iso_forest = IsolationForest(contamination=0.01, random_state=42)
    anomalies = iso_forest.fit_predict(X)
    
    # -1 is anomaly, 1 is normal
    n_anomalies = (anomalies == -1).sum()
    print(f"Detected {n_anomalies} anomalies out of {len(X)} samples.")
    return iso_forest, anomalies

def train_regression(X, y):
    print("\n--- Regression (Ridge) ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)
    
    y_pred = ridge.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"RMSE: {rmse:.6f}")
    print(f"R2 Score: {r2:.4f}")
    return ridge

def main():
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    print("Loading data...")
    df = load_and_prepare_data()
    
    # Feature Selection
    # We use volatility, market context, and time features
    features = ['volatility_24h', 'btc_close', 'eth_close', 'corr_btc_24h', 'corr_eth_24h', 'hour', 'day_of_week']
    
    # Encode symbol if we want to use it, but for now let's keep it general or drop it.
    # Let's stick to numerical features for simplicity and generalization.
    
    X = df[features]
    
    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 1. Clustering
    kmeans, clusters = train_clustering(X_scaled)
    df['cluster'] = clusters
    
    # 2. Classification
    # Target: stability_label
    y_class = df['stability_label']
    rf_model = train_classification(X_scaled, y_class)
    
    # 3. Anomaly Detection
    # We use the same features to find anomalous market states
    iso_model, anomalies = train_anomaly_detection(X_scaled)
    df['anomaly'] = anomalies
    
    # 4. Regression
    # Target: Next hour's deviation (shift -1)
    # We need to be careful with shifting per symbol
    df['target_deviation'] = df.groupby('symbol')['deviation_pct'].shift(-1)
    
    # Drop last row per symbol (NaN target)
    df_reg = df.dropna(subset=['target_deviation'])
    X_reg = df_reg[features]
    X_reg_scaled = scaler.transform(X_reg) # Use same scaler
    y_reg = df_reg['target_deviation']
    
    ridge_model = train_regression(X_reg_scaled, y_reg)
    
    print("\nModeling Complete.")
    
    # Save results sample
    df.to_csv(os.path.join(RESULTS_DIR, 'modeling_results.csv'), index=False)
    print(f"Saved results to {RESULTS_DIR}/modeling_results.csv")

if __name__ == "__main__":
    main()
