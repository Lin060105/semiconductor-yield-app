from pycaret.classification import *
import pandas as pd
import os

# 設定 matplotlib 字型 (避免中文亂碼，選用)
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False

print("--- Step 1: Loading Data & Setting up Environment ---")
# 讀取資料
data = pd.read_csv('../data/secom_processed.csv')

# 初始化環境 (跟之前一樣)
# session_id=123 確保結果可重現
s = setup(
    data=data, 
    target='label', 
    session_id=123, 
    normalize=True, 
    fix_imbalance=True, 
    verbose=False, 
    html=False
)
print("PyCaret environment initialized.")

print("\n--- Step 2: Training a Tree-Based Model (Random Forest) ---")
# 【關鍵改變】我們不讓 AutoML 自己選，我們直接指定要訓練 'rf' (Random Forest)
# 因為 Random Forest 是樹狀模型，完美支援 SHAP 和我們後面的 App
print("Training Random Forest model... (This is better for SHAP)")
rf_model = create_model('rf', verbose=False)
print("Random Forest model trained.")

print("\n--- Step 3: Saving the New Model ---")
# 我們把這個新模型存檔，覆蓋掉原本那個 Ridge 模型
# 這樣之後您的 App 就會用到這個更強的模型了
save_model(rf_model, '../output/final_yield_prediction_model')
print("Model overwritten with Random Forest.")

print("\n--- Step 4: Generating SHAP Plots ---")
plot_output_dir = '../output/shap_plots'
if not os.path.exists(plot_output_dir):
    os.makedirs(plot_output_dir)

print("Generating SHAP Summary Plot...")
try:
    # 這次一定會成功，因為 rf_model 是樹狀模型
    interpret_model(rf_model, plot='summary', save=True)
    
    # 搬移圖片
    if os.path.exists('Summary Plot.png'):
        target_file = os.path.join(plot_output_dir, 'shap_summary_plot.png')
        if os.path.exists(target_file):
            os.remove(target_file)
        os.rename('Summary Plot.png', target_file)
        print(f" -> Success! Plot saved to {target_file}")
except Exception as e:
    print(f"Error: {e}")

print("-" * 30)
print("All done! You are ready for Level 4.")