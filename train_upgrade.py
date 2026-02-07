from pycaret.classification import *
import pandas as pd
import pickle
import os

print("🚀 開始執行模型升級程序...")

# 1. 載入資料
print("📦 正在載入資料...")
if not os.path.exists('secom_processed.csv'):
    raise FileNotFoundError("找不到 secom_processed.csv，請確認檔案在同一個目錄下！")
    
dataset = pd.read_csv('secom_processed.csv')

# 2. 生成特徵清單 (解決 app.py 依賴問題)
print("📝 正在生成特徵清單 (required_features.pkl)...")
required_features = dataset.drop('label', axis=1).columns.tolist()
with open('required_features.pkl', 'wb') as f:
    pickle.dump(required_features, f)
print("✅ 特徵清單已儲存！")

# 3. 設定 PyCaret 環境 (關鍵：加入 fix_imbalance 來提升 Recall)
print("⚙️ 設定訓練環境 (啟用 SMOTE)...")
# session_id 固定為 123 確保結果可重現
# fix_imbalance=True 會自動處理 0/1 樣本不均的問題
s = setup(data=dataset, target='label', session_id=123, 
          fix_imbalance=True, verbose=False)

# 4. 訓練 Random Forest 模型
print("🌲 正在訓練 Random Forest 模型...")
rf = create_model('rf', verbose=False)

# 5. 存檔
print("💾 正在儲存最終模型...")
final_rf = finalize_model(rf)
save_model(final_rf, 'final_yield_prediction_model')

print("\n🎉 升級完成！")
print("1. 新模型已儲存為: final_yield_prediction_model.pkl")
print("2. 系統檔案已儲存為: required_features.pkl")
print("現在你可以更新 app.py 了。")