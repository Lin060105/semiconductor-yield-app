from pycaret.classification import *
import pandas as pd
import pickle
import os
import shutil
import matplotlib.pyplot as plt

# 設定 Matplotlib 後端
plt.switch_backend('Agg')

print("🚀 開始執行模型升級與報告生成程序 (v3.0 多模型比較版)...")

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
# log_experiment=True 可以記錄實驗，但這裡我們保持簡單
s = setup(data=dataset, target='label', session_id=123, 
          fix_imbalance=True, verbose=False)

# --- 4. 訓練與比較模型 ---
print("🏎️ 正在比較模型 (Random Forest, XGBoost, LightGBM)...")
# include 參數指定我們要比較的模型 ID
# sort='AUC' 表示我們依據 AUC 來選擇最佳模型 (針對不平衡資料集 AUC 通常比 Accuracy 好)
best_model = compare_models(include=['rf', 'xgboost', 'lightgbm'], sort='AUC', verbose=False)

# 抓取比較結果表
comparison_results = pull()
comparison_csv_path = os.path.join(REPORT_DIR, 'model_comparison.csv')
comparison_results.to_csv(comparison_csv_path)
print(f"   -> 🏆 最佳模型已選擇: {best_model}")
print(f"   -> 📄 模型比較報表已儲存至: {comparison_csv_path}")

# --- 5. 生成評估報告 ---
print("📊 正在生成最佳模型的評估圖表...")
plots = {
    'confusion_matrix': 'Confusion Matrix.png',
    'auc': 'AUC.png',
    'feature': 'Feature Importance.png',
}

for plot_type, file_name in plots.items():
    try:
        # 清除之前的圖表
        plt.clf()
        plot_model(best_model, plot=plot_type, save=True)
        
        # PyCaret save=True 會存成 'Confusion Matrix.png' (檔名可能有空格)
        # 我們需要確保將其移動到 reports 資料夾
        if os.path.exists(file_name):
            target_path = os.path.join(REPORT_DIR, file_name)
            if os.path.exists(target_path):
                os.remove(target_path)
            shutil.move(file_name, target_path)
            print(f"   -> 已儲存 {file_name}")
        else:
            print(f"   ⚠️ PyCaret 未生成預期檔名 {file_name}，可能已直接存入目錄或檔名不同。")
            
    except Exception as e:
        print(f"   ⚠️ 無法生成 {file_name}: {e}")

# --- 6. 生成 SHAP 解釋圖 (手動強制存檔) ---
print("🧠 正在計算 SHAP Values (使用 Matplotlib 強制存檔)...")
try:
    # 清除畫布
    plt.close('all')
    plt.figure(figsize=(10, 8))
    
    # 針對 Tree-based model (RF, XGB, LGBM) 進行解釋
    interpret_model(best_model, plot='summary', save=False)
    
    shap_dest = os.path.join(REPORT_DIR, 'SHAP Summary.png')
    plt.savefig(shap_dest, bbox_inches='tight', dpi=300)
    plt.close()
    
    if os.path.exists(shap_dest):
        print(f"   -> ✅ SHAP Summary 已手動成功儲存至 {shap_dest}")
    else:
        print("   ❌ 存檔失敗，請檢查權限。")

except Exception as e:
    print(f"   ❌ SHAP 生成失敗: {e}")
    print("      (提示: XGBoost/LightGBM 的 SHAP 支援通常良好，若失敗請檢查 shap 版本)")

# --- 7. 存檔 ---
print("💾 正在儲存最佳模型...")
final_model = finalize_model(best_model)
save_model(final_model, 'final_yield_prediction_model')
# 同步複製一份到 reports 供備份或下載
shutil.copy('final_yield_prediction_model.pkl', os.path.join(REPORT_DIR, 'final_yield_prediction_model.pkl'))

print("\n🎉 階段2-步驟1 執行完成！模型比較與升級結束。")