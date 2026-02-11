import pytest
import pandas as pd
import numpy as np
import os
import pickle
from pycaret.classification import load_model, predict_model

# --- 設定測試環境 ---
MODEL_PATHS = [
    'output/final_yield_prediction_model',
    '../output/final_yield_prediction_model',
    'final_yield_prediction_model',
    'reports/final_yield_prediction_model'
]

FEATURE_LIST_PATH = 'required_features.pkl'

def get_model_path():
    for path in MODEL_PATHS:
        if os.path.exists(path + '.pkl'):
            return path
    return None

def get_required_features():
    """嘗試載入訓練時保存的特徵清單，若找不到則使用預設值"""
    # 嘗試多個路徑尋找 pickle
    paths = [FEATURE_LIST_PATH, '../' + FEATURE_LIST_PATH, 'reports/' + FEATURE_LIST_PATH]
    
    for path in paths:
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    features = pickle.load(f)
                    print(f"✅ Loaded features from {path}")
                    return features
            except Exception as e:
                print(f"⚠️ Error loading features: {e}")
    
    # [Fallback] 如果真的找不到檔案，根據錯誤訊息，模型需要 feature_1 ~ feature_590
    # 這是最後的保險手段
    print("⚠️ Warning: required_features.pkl not found. Using fallback 'feature_X' naming.")
    return [f'feature_{i}' for i in range(1, 591)]

# --- Pytest Fixtures ---
@pytest.fixture(scope="module")
def model():
    path = get_model_path()
    if path is None:
        pytest.skip("❌ Model file not found. Skipping prediction tests.")
    return load_model(path)

@pytest.fixture(scope="module")
def feature_columns():
    """取得模型需要的正確欄位名稱"""
    return get_required_features()

@pytest.fixture
def single_input_data(feature_columns):
    """產生一筆模擬的單點輸入資料 (全為 0)，欄位名稱正確"""
    data = {col: [0.0] for col in feature_columns}
    return pd.DataFrame(data)

@pytest.fixture
def batch_input_data(feature_columns):
    """產生一批模擬的輸入資料 (10 筆隨機數值)，欄位名稱正確"""
    np.random.seed(42)
    num_rows = 10
    # 產生隨機矩陣
    data = np.random.rand(num_rows, len(feature_columns))
    df = pd.DataFrame(data, columns=feature_columns)
    return df

# --- Test Cases ---

def test_model_type(model):
    assert model is not None, "Model failed to load"
    print("\n✅ Model loaded successfully")

def test_single_prediction_format(model, single_input_data):
    """測試 2: 單點預測"""
    try:
        predictions = predict_model(model, data=single_input_data)
        
        # 檢查欄位 (PyCaret 3.0+ 輸出 prediction_label 和 prediction_score)
        cols = predictions.columns
        has_label = 'prediction_label' in cols or 'Label' in cols
        has_score = 'prediction_score' in cols or 'Score' in cols
        
        assert has_label, f"Missing prediction label. Columns: {cols}"
        assert has_score, f"Missing prediction score. Columns: {cols}"
        print("\n✅ Single prediction format check passed")
    except Exception as e:
        pytest.fail(f"Prediction failed with error: {str(e)}")

def test_batch_prediction_shape(model, batch_input_data):
    """測試 3: 批次預測"""
    try:
        predictions = predict_model(model, data=batch_input_data)
        assert len(predictions) == 10
        print("\n✅ Batch prediction shape check passed")
    except Exception as e:
         pytest.fail(f"Batch prediction failed: {e}")

def test_prediction_values_range(model, batch_input_data):
    """測試 4: 數值範圍檢查"""
    predictions = predict_model(model, data=batch_input_data)
    
    if 'prediction_score' in predictions.columns:
        scores = predictions['prediction_score']
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0
        print("\n✅ Prediction score range check passed")