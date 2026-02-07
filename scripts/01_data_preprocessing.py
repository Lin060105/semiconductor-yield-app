import pandas as pd
import numpy as np

print("--- Step 1: Loading Data ---")
# 讀取特徵和標籤數據
features_path = '../data/secom_features.txt'
labels_path = '../data/secom_labels.txt'

# 自動生成欄位名稱 feature_1 到 feature_590
feature_names = [f'feature_{i+1}' for i in range(590)]

# 讀取檔案，sep='\s+' 代表用空格分隔
df_features = pd.read_csv(features_path, sep='\s+', header=None, names=feature_names)
df_labels = pd.read_csv(labels_path, sep='\s+', header=None, names=['label', 'timestamp'])

# 合併特徵和標籤
df_raw = pd.concat([df_features, df_labels], axis=1)
# 移除不需要的時間戳記
df_raw = df_raw.drop('timestamp', axis=1)

print(f"Raw data loaded. Shape: {df_raw.shape}")
print("-" * 30)

print("\n--- Step 2: Handling Missing Values ---")
# 移除那些整欄都是空的特徵
df_cleaned = df_raw.dropna(axis=1, how='all')
#剩下的缺失值用平均值填補
df_imputed = df_cleaned.fillna(df_cleaned.mean())
print(f"Shape after cleaning: {df_imputed.shape}")
print("-" * 30)

print("\n--- Step 3: Removing Zero-Variance Features ---")
# 移除數值完全沒變化的特徵 (對預測沒幫助)
zero_variance_cols = df_imputed.columns[df_imputed.nunique() == 1]
df_processed = df_imputed.drop(columns=zero_variance_cols)
print(f"Removed {len(zero_variance_cols)} zero-variance columns.")
print(f"Shape after removing zero-variance columns: {df_processed.shape}")
print("-" * 30)

print("\n--- Step 4: Final Data Preparation and Saving ---")
# 轉換標籤: -1 (Pass) 改為 0, 1 (Fail) 改為 1
df_processed['label'] = df_processed['label'].replace({-1: 0, 1: 1})

# 存成一個處理好的 CSV 檔
output_path = '../data/secom_processed.csv'
df_processed.to_csv(output_path, index=False)
print(f"Preprocessing complete. Processed data saved to: {output_path}")
print("-" * 30)