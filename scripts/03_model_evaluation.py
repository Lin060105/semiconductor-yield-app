from pycaret.classification import *
import pandas as pd
import os
import matplotlib.pyplot as plt

print("--- Step 1: Loading Data ---")
# 讀取資料
data_path = '../data/secom_processed.csv'
dataset = pd.read_csv(data_path)

# 【關鍵修正】
# PyCaret 3.x 在畫圖前必須先執行 setup() 來初始化環境
# 我們使用跟訓練時一樣的設定，這樣 plot_model 才知道怎麼處理數據
print("--- Step 2: Initializing PyCaret Context ---")
s = setup(data=dataset, target='label', session_id=123, verbose=False, html=False)
print("PyCaret environment initialized.")

print("\n--- Step 3: Loading the Saved Model ---")
# 載入剛剛訓練好的模型
model_path = '../output/final_yield_prediction_model'
saved_model = load_model(model_path)
print("Model loaded successfully.")
print("-" * 30)

print("\n--- Step 4: Making Predictions ---")
# 我們切出最後 10% 的資料來當作「考題」（模擬測試集）
test_df = dataset.sample(frac=0.1, random_state=123)
X_test = test_df.drop('label', axis=1) # 題目
y_test = test_df['label'] # 答案

# 讓模型作答
predictions = predict_model(saved_model, data=test_df)
print("Prediction on test set complete.")
print(predictions.head())
print("-" * 30)

print("\n--- Step 5: Generating Evaluation Plots ---")
# 建立存放圖表的資料夾
plot_output_dir = '../output/automl_reports'
if not os.path.exists(plot_output_dir):
    os.makedirs(plot_output_dir)

print(f"Saving plots to {plot_output_dir}...")

# 1. 混淆矩陣 (看它有沒有把壞品誤判為良品)
print("Generating Confusion Matrix...")
try:
    plot_model(saved_model, plot='confusion_matrix', save=True)
    # PyCaret 存圖會存在當前目錄，我們把它搬到 output 資料夾
    if os.path.exists('Confusion Matrix.png'):
        if os.path.exists(os.path.join(plot_output_dir, 'confusion_matrix.png')):
            os.remove(os.path.join(plot_output_dir, 'confusion_matrix.png'))
        os.rename('Confusion Matrix.png', os.path.join(plot_output_dir, 'confusion_matrix.png'))
except Exception as e:
    print(f"Could not generate Confusion Matrix: {e}")

# 2. AUC 曲線 (看模型的鑑別力)
print("Generating AUC Curve...")
try:
    plot_model(saved_model, plot='auc', save=True)
    if os.path.exists('AUC.png'):
        if os.path.exists(os.path.join(plot_output_dir, 'auc_roc_curve.png')):
            os.remove(os.path.join(plot_output_dir, 'auc_roc_curve.png'))
        os.rename('AUC.png', os.path.join(plot_output_dir, 'auc_roc_curve.png'))
except Exception as e:
    print(f"Could not generate AUC Curve: {e}")

# 3. 特徵重要性 (看哪些感測器最關鍵)
print("Generating Feature Importance...")
try:
    plot_model(saved_model, plot='feature', save=True)
    if os.path.exists('Feature Importance.png'):
        if os.path.exists(os.path.join(plot_output_dir, 'feature_importance.png')):
            os.remove(os.path.join(plot_output_dir, 'feature_importance.png'))
        os.rename('Feature Importance.png', os.path.join(plot_output_dir, 'feature_importance.png'))
except Exception as e:
    print(f"Could not generate Feature Importance: {e}")

print("\nAll evaluation plots have been generated and saved.")
print("-" * 30)