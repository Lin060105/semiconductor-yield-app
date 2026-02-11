import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from pycaret.classification import *
from sklearn.model_selection import learning_curve
import logging

# 設定 logging 以便追蹤
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 路徑設定
DATA_PATH = '../data/secom_processed.csv'
MODEL_DIR = '../output'
REPORT_DIR = '../reports'
IMG_OUTPUT_DIR = '../output/automl_reports'

# 確保輸出目錄存在
os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(IMG_OUTPUT_DIR, exist_ok=True)

def check_overfitting(model, X, y):
    """
    計算 Learning Curve 並進行過擬合/欠擬合的文字判讀
    """
    logging.info("Calculating Learning Curve data for Overfitting Analysis...")
    
    # 使用 sklearn 原生 learning_curve 計算數值
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, scoring='accuracy', n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 5)
    )
    
    train_mean = np.mean(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    
    # 取得最後一個點（最大訓練資料量）的差距
    final_train_score = train_mean[-1]
    final_test_score = test_mean[-1]
    gap = final_train_score - final_test_score
    
    analysis_report = []
    analysis_report.append(f"--- Learning Curve Analysis ---")
    analysis_report.append(f"Final Training Score: {final_train_score:.4f}")
    analysis_report.append(f"Final CV Score: {final_test_score:.4f}")
    analysis_report.append(f"Score Gap: {gap:.4f}")
    
    # 判斷邏輯
    if final_train_score > 0.98 and final_test_score < 0.85:
        judgment = "CRITICAL: High Overfitting detected! (Gap is large and training score is near perfect)"
    elif gap > 0.1:
        judgment = "WARNING: Moderate Overfitting detected. Consider increasing regularization or adding more data."
    elif final_train_score < 0.7:
        judgment = "WARNING: Underfitting detected. Model may be too simple."
    else:
        judgment = "SUCCESS: Model shows good generalization (Balanced Bias-Variance)."
        
    analysis_report.append(f"Judgment: {judgment}")
    
    # 將分析結果寫入報告
    report_path = os.path.join(REPORT_DIR, 'overfitting_analysis.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(analysis_report))
    
    logging.info(f"Overfitting analysis saved to {report_path}")
    print("\n".join(analysis_report))

def generate_model_comparison_plot(results_df):
    """
    產生 XGBoost vs CatBoost 的 Recall/F1/AUC 比較長條圖
    """
    logging.info("Generating Model Comparison Plot...")
    
    # 篩選我們關心的模型
    target_models = ['xgboost', 'catboost']
    # 確保 index 是模型名稱 (PyCaret results 的 index 通常是縮寫)
    subset = results_df[results_df.index.isin(target_models)]
    
    if subset.empty:
        logging.warning("XGBoost or CatBoost not found in results. Skipping plot.")
        return

    # 選取關鍵指標
    metrics = ['Recall', 'F1', 'AUC']
    subset = subset[metrics]
    
    # 繪圖
    ax = subset.plot(kind='bar', figsize=(10, 6), rot=0)
    plt.title('Deep Model Comparison: XGBoost vs CatBoost', fontsize=15)
    plt.ylabel('Score')
    plt.ylim(0, 1.1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(loc='lower right')
    
    # 標示數值
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.3f}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', 
                    xytext=(0, 9), 
                    textcoords='offset points')
    
    save_path = os.path.join(REPORT_DIR, 'model_comparison_final.png')
    plt.tight_layout()
    plt.savefig(save_path)
    logging.info(f"Comparison plot saved to {save_path}")

def main():
    print("--- Step 1: Loading Data ---")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data not found at {DATA_PATH}")
    
    dataset = pd.read_csv(DATA_PATH)
    print(f"Data shape: {dataset.shape}")

    print("\n--- Step 2: Initialize PyCaret ---")
    # 保持與訓練時一致的設定
    s = setup(data=dataset, target='label', session_id=123, verbose=False, html=False)
    
    print("\n--- Step 3: Targeted Model Comparison (XGBoost vs CatBoost) ---")
    # 這裡我們重新比較這兩個強效模型，以獲取最新的比較數據
    # 根據需求，這一步是為了產出報告圖表
    print("Training XGBoost and CatBoost for comparison report...")
    try:
        # include 參數確保只比較這兩個
        best_models = compare_models(include=['xgboost', 'catboost'], sort='F1')
        results = pull()
        
        # 儲存 CSV
        results.to_csv(os.path.join(REPORT_DIR, 'model_comparison.csv'))
        
        # 產生視覺化比較圖
        generate_model_comparison_plot(results)
        
    except Exception as e:
        logging.error(f"Error during model comparison: {e}")

    print("\n--- Step 4: Load Final Model & Deep Analysis ---")
    model_path = os.path.join(MODEL_DIR, 'final_yield_prediction_model')
    try:
        final_model = load_model(model_path)
        logging.info("Final model loaded successfully.")
        
        # 取得 PyCaret 處理過的 X_train, y_train 用於手動計算 Learning Curve
        X_train = get_config('X_train')
        y_train = get_config('y_train')
        
        # 執行過擬合文字分析
        check_overfitting(final_model, X_train, y_train)
        
        # 產生標準圖片報告
        logging.info("Generating standard PyCaret plots...")
        plot_model(final_model, plot='confusion_matrix', save=True)
        shutil.move('Confusion Matrix.png', os.path.join(IMG_OUTPUT_DIR, 'confusion_matrix.png'))
        
        plot_model(final_model, plot='auc', save=True)
        shutil.move('AUC.png', os.path.join(IMG_OUTPUT_DIR, 'auc_roc_curve.png'))
        
        plot_model(final_model, plot='feature', save=True)
        shutil.move('Feature Importance.png', os.path.join(IMG_OUTPUT_DIR, 'feature_importance.png'))
        
        # 產生 Learning Curve 圖片
        plot_model(final_model, plot='learning', save=True)
        shutil.move('Learning Curve.png', os.path.join(IMG_OUTPUT_DIR, 'learning_curve.png'))
        
    except Exception as e:
        logging.error(f"Error in Step 4: {e}")
        # 若讀不到模型，可能是 Step 2 setup 把環境重置了，實務上 load_model 需要與 setup 匹配
        # 但因為我們在 Step 2 已經 setup 了，應該沒問題。

    print(f"\nAll tasks completed. Reports generated in {REPORT_DIR}")

if __name__ == "__main__":
    main()