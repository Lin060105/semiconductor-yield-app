import pandas as pd
import os
import matplotlib.pyplot as plt
from pycaret.classification import *

# 設定繪圖後端 (避免在無視窗環境報錯)
plt.switch_backend('Agg') 

def find_csv_file(filename='secom_processed.csv'):
    """暴力搜尋：從當前目錄往上找，直到找到檔案為止"""
    current_dir = os.getcwd()
    print(f"🔍 正在 {current_dir} 與其子資料夾中搜尋 {filename}...")
    
    # 1. 先找當前目錄
    if os.path.exists(filename):
        return os.path.abspath(filename)
    
    # 2. 遞迴搜尋 (往下找 3 層)
    for root, dirs, files in os.walk(current_dir):
        if filename in files:
            return os.path.join(root, filename)
            
    return None

def main():
    print("🚀 程式啟動...")

    # 1. 自動搜尋檔案
    csv_path = find_csv_file('secom_processed.csv')
    
    if not csv_path:
        print("\n❌ 找不到 secom_processed.csv！")
        print("請確認您有把 csv 檔案放在這個資料夾(或是子資料夾)裡面。")
        return

    print(f"✅ 找到檔案：{csv_path}")
    dataset = pd.read_csv(csv_path)

    # 2. 自動偵測 Target 欄位 (抓最後一欄)
    # 這裡修正了之前一直寫死 'label' 的錯誤
    target_col = dataset.columns[-1] 
    print(f"🎯 自動鎖定目標欄位：'{target_col}'")

    # 3. 初始化 PyCaret
    print("⚙️ 正在初始化環境 (Setup)...")
    try:
        s = setup(data=dataset, target=target_col, session_id=123, fix_imbalance=True, verbose=False)
    except Exception as e:
        print(f"❌ Setup 初始化失敗: {e}")
        return

    # 4. 訓練模型
    print("⏳ 正在訓練 Random Forest 模型 (請稍候)...")
    rf = create_model('rf', verbose=False)
    
    # 建立 reports 資料夾
    reports_dir = os.path.join(os.getcwd(), 'reports')
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)

    # 5. 存圖與移動
    def save_and_move(model, plot_type, filename):
        try:
            print(f"   -> 繪製 {filename}...")
            # 產生圖片 (PyCaret 會存在當前目錄)
            plot_model(model, plot=plot_type, save=True)
            
            # 處理檔名 (PyCaret 預設檔名 -> 我們要的檔名)
            default_map = {
                'confusion_matrix': 'Confusion Matrix.png',
                'auc': 'AUC.png',
                'feature': 'Feature Importance.png'
            }
            src_name = default_map.get(plot_type)
            
            # 搬移檔案
            if src_name and os.path.exists(src_name):
                dst_path = os.path.join(reports_dir, filename)
                if os.path.exists(dst_path):
                    os.remove(dst_path)
                os.rename(src_name, dst_path)
                print(f"      ✅ 已儲存至 reports/{filename}")
        except Exception as e:
            print(f"      ⚠️ 無法儲存 {filename}: {e}")

    print("📊 生成圖表中...")
    save_and_move(rf, 'confusion_matrix', 'confusion_matrix.png')
    save_and_move(rf, 'auc', 'auc_curve.png')
    save_and_move(rf, 'feature', 'feature_importance.png')

    # 6. 輸出數據
    results = pull()
    results.to_csv(os.path.join(reports_dir, 'model_metrics.csv'), index=False)
    
    print(f"\n✅ 全部完成！請打開 {reports_dir} 資料夾查看報告圖片。")

if __name__ == "__main__":
    main()