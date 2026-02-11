from pycaret.classification import *
import pandas as pd
import pickle
import os
import shutil
import matplotlib.pyplot as plt

# 設定 Matplotlib 後端，避免在無介面伺服器執行時報錯
plt.switch_backend('Agg')

print("🚀 開始執行模型升級與報告生成程序 (v4.0 專業多模型版)...")

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
print("⚙️ 設定訓練環境 (處理不平衡資料)...")
# fix_imbalance=True 使用 SMOTE 處理良率不平衡問題
s = setup(data=dataset, target='label', session_id=123, 
          fix_imbalance=True, verbose=False)

# --- 4. 訓練與比較模型 (RF, XGBoost, LightGBM, CatBoost) ---
print("🏎️ 正在比較模型 (Random Forest, XGBoost, LightGBM, CatBoost)...")
# 根據 Grok 建議，我們鎖定 Recall 與 F1 作為主要參考，因為半導體失效檢測更看重漏檢率
best_model = compare_models(
    include=['rf', 'xgboost', 'lightgbm', 'catboost'], 
    sort='Recall',  # 優先保證能抓出失敗樣品
    verbose=False
)

# 抓取比較結果表並儲存
comparison_results = pull()
comparison_csv_path = os.path.join(REPORT_DIR, 'model_comparison.csv')
comparison_results.to_csv(comparison_csv_path)
print(f"   -> 🏆 最佳模型已選擇: {best_model}")
print(f"   -> 📄 模型比較報表已儲存至: {comparison_csv_path}")

# --- 5. 生成評估報告 (含學習曲線，解決 Grok 提到的弱點) ---
print("📊 正在生成最佳模型的評估圖表...")
plots = {
    'confusion_matrix': 'Confusion Matrix.png',
    'auc': 'AUC.png',
    'feature': 'Feature Importance.png',
    'learning': 'Learning Curve.png', # 新增學習曲線檢查過擬合
    'pr': 'Precision Recall.png'     # 新增 PR 曲線針對不平衡資料
}

for plot_type, file_name in plots.items():
    try:
        plt.clf()
        plot_model(best_model, plot=plot_type, save=True)
        
        # 處理 PyCaret 存檔名稱中的空格與路徑移動
        generated_file = f"{plot_type.capitalize()}.png" if plot_type != 'confusion_matrix' else 'Confusion Matrix.png'
        if os.path.exists(generated_file):
            target_path = os.path.join(REPORT_DIR, file_name)
            if os.path.exists(target_path):
                os.remove(target_path)
            shutil.move(generated_file, target_path)
            print(f"   -> 已儲存 {file_name}")
    except Exception as e:
        print(f"   ⚠️ 無法生成 {file_name}: {e}")

# --- 6. 生成 SHAP 解釋圖 ---
print("🧠 正在計算 SHAP Values...")
try:
    plt.close('all')
    interpret_model(best_model, plot='summary', save=True)
    
    # interpret_model 的 save=True 通常存為 'SHAP Summary.png'
    if os.path.exists('SHAP Summary.png'):
        shutil.move('SHAP Summary.png', os.path.join(REPORT_DIR, 'SHAP Summary.png'))
        print(f"   -> ✅ SHAP Summary 儲存完成")
except Exception as e:
    print(f"   ❌ SHAP 生成失敗: {e}")

# --- 7. 最終模型存檔 ---
print("💾 正在儲存最佳模型...")
final_model = finalize_model(best_model)
save_model(final_model, 'final_yield_prediction_model')
shutil.copy('final_yield_prediction_model.pkl', os.path.join(REPORT_DIR, 'final_yield_prediction_model.pkl'))

print("\n🎉 階段 2 步驟 1 執行完成！已完成多模型比較與學習曲線生成。")