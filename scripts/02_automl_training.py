import pandas as pd
from pycaret.classification import *
import os

print("--- Step 1: Loading Processed Data ---")
# 載入剛剛處理好的資料
data_path = '../data/secom_processed.csv'
dataset = pd.read_csv(data_path)
print(f"Data loaded successfully. Shape: {dataset.shape}")
print("-" * 30)

print("\n--- Step 2: Setting up PyCaret Environment ---")
# 建立輸出資料夾，以免報錯
if not os.path.exists('../output/automl_reports'):
    os.makedirs('../output/automl_reports')

# 初始化 PyCaret 設定
# 這邊會自動做特徵標準化(normalize)和處理類別不平衡(fix_imbalance)
clf_session = setup(
    data=dataset,
    target='label',
    session_id=123,
    normalize=True,
    fix_imbalance=True,
    html=False,  # 關閉瀏覽器跳出
    verbose=False # 減少雜訊輸出
)
print("PyCaret setup complete.")
print("-" * 30)

print("\n--- Step 3: Comparing Machine Learning Models ---")
# 這是最神奇的一行：自動比較多種模型
print("Training and comparing models... (This may take a few minutes)")
# 我們依據 'F1' 分數來排名，因為良率預測通常更在乎抓出壞品
best_model = compare_models(sort='F1')
print("Model comparison complete.")

# 顯示前幾名的結果
results = pull()
print("Top models performance:")
print(results.head())
print("-" * 30)

print("\n--- Step 4: Finalizing and Saving the Model ---")
# 使用全部資料重新訓練最佳模型
final_model = finalize_model(best_model)

# 儲存模型
model_path = '../output/final_yield_prediction_model'
save_model(final_model, model_path)
print(f"Model saved successfully to {model_path}.pkl")
print("-" * 30)