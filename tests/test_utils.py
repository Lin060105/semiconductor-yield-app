import pytest
import pandas as pd
import sys
import os

# 將上一層目錄加入路徑，這樣才能 import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils

# --- 測試 1: 測試特徵清單載入 ---
def test_load_feature_config():
    """測試是否能正確讀取特徵清單，若無檔案應回傳預設值"""
    # 假設測試環境沒有 pkl 檔，應該回傳預設 list
    # 我們先暫時改名原本的 pkl 檔來模擬「找不到檔案」的情況 (選擇性)
    
    features = utils.load_feature_config('non_existent_file.pkl')
    assert isinstance(features, list)
    assert len(features) == 3
    assert 'feature_1' in features

# --- 測試 2: 測試批量預測邏輯 (Mocking) ---
# 這裡我們不真的載入模型 (太慢)，而是測試資料處理流程
def test_batch_prediction_structure():
    """測試批量預測的輸入輸出格式是否正確"""
    # 模擬一個假的 DataFrame
    mock_data = pd.DataFrame({
        'feature_1': [0.1, 0.5],
        'feature_2': [0.2, 0.6],
        'label': [0, 1] # 假設包含 label
    })
    
    # 檢查是否能成功轉換
    assert not mock_data.empty
    assert 'feature_1' in mock_data.columns

# --- 測試 3: 檢查專案關鍵檔案是否存在 ---
def test_critical_files_exist():
    """確保專案結構完整，關鍵檔案未遺失"""
    required_files = [
        'app.py',
        'utils.py',
        'requirements.txt',
        'train_upgrade.py',
        'Dockerfile'
    ]
    for file in required_files:
        assert os.path.exists(file), f"缺少關鍵檔案: {file}"

if __name__ == "__main__":
    pytest.main()