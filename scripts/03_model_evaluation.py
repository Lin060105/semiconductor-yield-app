from pycaret.classification import *
import pandas as pd
import os
import shutil

# 設定報告輸出路徑
REPORT_OUTPUT_DIR = '../reports' # 改為存到 reports 資料夾，符合 GitHub 結構建議
IMG_OUTPUT_DIR = '../output/automl_reports'

# 確保資料夾存在
os.makedirs(REPORT_OUTPUT_DIR, exist_ok=True)
os.makedirs(IMG_OUTPUT_DIR, exist_ok=True)

print("--- Step 1: Loading Data ---")
# 讀取資料
data_path = '../data/secom_processed.csv'
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Data file not found at {data_path}. Please run 01_data_preprocessing.py first.")

dataset = pd.read_csv(data_path)
print(f"Data loaded. Shape: {dataset.shape}")

print("--- Step 2: Initializing PyCaret Context ---")
# 初始化環境，確保與訓練時一致
s = setup(data=dataset, target='label', session_id=123, verbose=False, html=False)
print("PyCaret environment initialized.")

print("\n--- Step 3: Loading the Saved Model ---")
# 載入訓練好的模型
model_path = '../output/final_yield_prediction_model'
try:
    saved_model = load_model(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    # 如果讀不到模型，這裡可以選擇終止或重新訓練，這裡我們先終止
    raise e

print("-" * 30)

print("\n--- Step 4: Generating Model Comparison Report (Benchmark) ---")
# 為了證明我們的模型選擇是正確的，我們跑一個快速的比較
# 包含 XGBoost, LightGBM, Random Forest
print("Running comparison between XGBoost, LightGBM, and RF...")
try:
    # include 參數指定要比較的模型 ID
    # n_select=1 只要最好的，但我們主要是要那個表格
    best_benchmark = compare_models(include=['xgboost', 'lightgbm', 'rf'], n_select=1)
    
    # pull() 函式可以抓取最近一次執行的結果表格 (DataFrame)
    comparison_results = pull()
    
    # 儲存比較表格到 reports 資料夾
    csv_path = os.path.join(REPORT_OUTPUT_DIR, 'model_comparison_benchmark.csv')
    comparison_results.to_csv(csv_path)
    print(f"Model comparison table saved to: {csv_path}")
    print(comparison_results)
except Exception as e:
    print(f"Skipping model comparison due to error: {e}")

print("-" * 30)

print("\n--- Step 5: Generating Evaluation Plots ---")

def move_plot(filename, dest_name):
    """
    輔助函式：將 PyCaret 生成在根目錄的圖片移動到指定資料夾
    """
    try:
        if os.path.exists(filename):
            dest_path = os.path.join(IMG_OUTPUT_DIR, dest_name)
            if os.path.exists(dest_path):
                os.remove(dest_path)
            shutil.move(filename, dest_path)
            print(f"Saved: {dest_name}")
        else:
            print(f"Warning: {filename} was not generated.")
    except Exception as e:
        print(f"Error moving {filename}: {e}")

# 1. 混淆矩陣
print("Generating Confusion Matrix...")
try:
    plot_model(saved_model, plot='confusion_matrix', save=True)
    move_plot('Confusion Matrix.png', 'confusion_matrix.png')
except Exception as e:
    print(f"Could not generate Confusion Matrix: {e}")

# 2. AUC 曲線
print("Generating AUC Curve...")
try:
    plot_model(saved_model, plot='auc', save=True)
    move_plot('AUC.png', 'auc_roc_curve.png')
except Exception as e:
    print(f"Could not generate AUC Curve: {e}")

# 3. 特徵重要性
print("Generating Feature Importance...")
try:
    plot_model(saved_model, plot='feature', save=True)
    move_plot('Feature Importance.png', 'feature_importance.png')
except Exception as e:
    print(f"Could not generate Feature Importance: {e}")

# 4. [新增] 學習曲線 (Learning Curve) - 檢查過擬合
print("Generating Learning Curve (Overfitting Check)...")
try:
    # 學習曲線計算量較大，可能需要一點時間
    plot_model(saved_model, plot='learning', save=True)
    move_plot('Learning Curve.png', 'learning_curve.png')
except Exception as e:
    print(f"Could not generate Learning Curve: {e}")

print("-" * 30)
print("\nAll evaluation tasks completed.")
print(f"Reports saved in: {REPORT_OUTPUT_DIR}")
print(f"Images saved in: {IMG_OUTPUT_DIR}")