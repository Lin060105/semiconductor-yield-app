from pycaret.classification import *
import pandas as pd
import pickle
import os
import shutil
import matplotlib.pyplot as plt

# 設定 Matplotlib 後端
plt.switch_backend('Agg')

print("🚀 開始執行模型升級與報告生成程序 (v2.3 手動存檔版)...")

# --- 0. 環境準備 ---
REPORT_DIR = 'reports'
if not os.path.exists(REPORT_DIR):
    os.makedirs(REPORT_DIR)

# --- 1. 載入資料 ---
print("📦 正在載入資料...")
DATA_FILE = 'secom_processed.csv'
if not os.path.exists(DATA_FILE):
    if os.path.exists(os.path.join('data', DATA_FILE)):
        DATA_FILE = os.path.join('data', DATA_FILE)
    else:
        raise FileNotFoundError(f"❌ 找不到 {DATA_FILE}")

dataset = pd.read_csv(DATA_FILE)

# --- 2. 生成特徵清單 ---
print("📝 正在生成特徵清單...")
required_features = dataset.drop('label', axis=1).columns.tolist()
with open('required_features.pkl', 'wb') as f:
    pickle.dump(required_features, f)

# --- 3. 設定 PyCaret 環境 ---
print("⚙️ 設定訓練環境...")
s = setup(data=dataset, target='label', session_id=123, 
          fix_imbalance=True, verbose=False)

# --- 4. 訓練模型 ---
print("🌲 正在訓練 Random Forest 模型...")
rf = create_model('rf', verbose=False)

# --- 5. 生成評估報告 ---
print("📊 正在生成基礎評估圖表...")
plots = {
    'confusion_matrix': 'Confusion Matrix.png',
    'auc': 'AUC.png',
    'feature': 'Feature Importance.png',
}

for plot_type, file_name in plots.items():
    try:
        # 清除之前的圖表
        plt.clf()
        plot_model(rf, plot=plot_type, save=True)
        
        if os.path.exists(file_name):
            if os.path.exists(os.path.join(REPORT_DIR, file_name)):
                os.remove(os.path.join(REPORT_DIR, file_name))
            shutil.move(file_name, os.path.join(REPORT_DIR, file_name))
            print(f"   -> 已儲存 {file_name}")
    except Exception as e:
        print(f"   ⚠️ 無法生成 {file_name}: {e}")

# --- 6. 生成 SHAP 解釋圖 (手動強制存檔) ---
print("🧠 正在計算 SHAP Values (使用 Matplotlib 強制存檔)...")
try:
    # 清除畫布，避免重疊
    plt.close('all')
    plt.figure(figsize=(10, 8))
    
    # 關鍵修改：save=False，讓它畫在我們建立的 plt 上
    interpret_model(rf, plot='summary', save=False)
    
    # 定義路徑
    shap_dest = os.path.join(REPORT_DIR, 'SHAP Summary.png')
    
    # 強制手動存檔
    plt.savefig(shap_dest, bbox_inches='tight', dpi=300)
    plt.close() # 關閉資源
    
    if os.path.exists(shap_dest):
        print(f"   -> ✅ SHAP Summary 已手動成功儲存至 {shap_dest}")
    else:
        print("   ❌ 存檔指令執行後仍找不到檔案，請檢查磁碟權限。")

except Exception as e:
    print(f"   ❌ SHAP 生成失敗: {e}")
    print("      (如果錯誤是 'module not found'，請執行 pip install shap==0.41.0)")

# --- 7. 存檔 ---
print("💾 正在儲存最終模型...")
final_rf = finalize_model(rf)
save_model(final_rf, 'final_yield_prediction_model')

print("\n🎉 修正版執行完成！")