import pytest
import pandas as pd
import os
import pickle
from pycaret.classification import load_model, predict_model

# --- 設定路徑 ---
# 測試環境可能在根目錄執行，也可能在 tests 目錄執行，這裡統一處理
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_NAME = 'final_yield_prediction_model'
FEATURE_LIST_FILE = 'required_features.pkl'

# --- Fixtures (測試前的準備工作) ---

@pytest.fixture(scope="module")
def model_path():
    """尋找模型檔案路徑"""
    # 優先找 reports 資料夾 (符合 app.py 邏輯)
    report_path = os.path.join(BASE_DIR, 'reports', MODEL_NAME)
    root_path = os.path.join(BASE_DIR, MODEL_NAME)
    
    if os.path.exists(report_path + '.pkl'):
        return report_path
    elif os.path.exists(root_path + '.pkl'):
        return root_path
    else:
        return None

@pytest.fixture(scope="module")
def feature_list():
    """載入特徵清單，用於建立假資料"""
    path = os.path.join(BASE_DIR, FEATURE_LIST_FILE)
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    else:
        return None

@pytest.fixture(scope="module")
def model(model_path):
    """載入 PyCaret 模型"""
    if model_path:
        return load_model(model_path)
    else:
        pytest.skip("⚠️ 跳過測試：找不到模型檔案 (.pkl)。請先執行 python train_upgrade.py")

# --- Test Cases (測試案例) ---

def test_environment_setup():
    """測試：基本環境與依賴庫是否正常"""
    try:
        import xgboost
        import lightgbm
        assert True
    except ImportError as e:
        pytest.fail(f"❌ 缺少必要套件: {e}")

def test_feature_list_exists(feature_list):
    """測試：required_features.pkl 是否存在"""
    assert feature_list is not None, "❌ 找不到特徵清單 (required_features.pkl)"
    assert len(feature_list) > 0, "❌ 特徵清單是空的"

def test_model_loading(model):
    """測試：模型是否能成功載入"""
    assert model is not None, "❌ 模型載入失敗"

def test_prediction_pipeline(model, feature_list):
    """測試：建立一筆假資料並執行預測 (Integration Test)"""
    # 1. 建立一筆全為 0 的 Dummy Data
    # 因為我們不知道特徵的實際範圍，填 0 是最安全的測試方法 (假設有做過 Imputation)
    dummy_data = pd.DataFrame([0] * len(feature_list)).T
    dummy_data.columns = feature_list
    
    # 2. 執行預測
    try:
        predictions = predict_model(model, data=dummy_data)
    except Exception as e:
        pytest.fail(f"❌ 預測執行崩潰: {e}")

    # 3. 驗證輸出欄位
    # PyCaret 不同版本輸出的欄位名可能不同 (Label/Score 或 prediction_label/prediction_score)
    cols = predictions.columns
    has_label = 'prediction_label' in cols or 'Label' in cols
    has_score = 'prediction_score' in cols or 'Score' in cols
    
    assert has_label, f"❌ 預測結果缺少標籤欄位。現有欄位: {cols}"
    assert has_score, f"❌ 預測結果缺少分數欄位。現有欄位: {cols}"

    print("\n✅ 預測管線測試通過！")