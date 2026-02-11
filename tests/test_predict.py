import pytest
import pandas as pd
import numpy as np
import os
import pickle
from pycaret.classification import load_model, predict_model

# --- 設定路徑 ---
MODEL_PATHS = [
    'output/final_yield_prediction_model',
    '../output/final_yield_prediction_model',
    'final_yield_prediction_model',
    'reports/final_yield_prediction_model'
]

DATA_PATHS = [
    'data/secom_processed.csv',
    '../data/secom_processed.csv',
    'reports/secom_processed.csv'
]

def get_model_path():
    """搜尋模型檔案是否存在"""
    for path in MODEL_PATHS:
        # load_model 不需要加 .pkl，但檢查存在時需要
        if os.path.exists(path + '.pkl'):
            return path
    return None

def get_required_features():
    """
    動態取得模型需要的特徵欄位名稱。
    策略：優先從訓練資料 (csv) 讀取標頭，確保與訓練時一致。
    """
    # 1. 嘗試從處理過的資料集讀取欄位 (最準確)
    for path in DATA_PATHS:
        if os.path.exists(path):
            try:
                # 只讀取 Header，不讀取整個檔案，速度快
                df_head = pd.read_csv(path, nrows=0)
                # 排除標籤欄位 'label'，剩下的就是特徵
                features = [c for c in df_head.columns if c != 'label']
                print(f"✅ Loaded {len(features)} features from {path}")
                return features
            except Exception as e:
                print(f"⚠️ Error reading features from CSV: {e}")

    # 2. [Fallback] 如果真的找不到資料，回傳空清單讓測試失敗或跳過，
    #    而不是給錯誤的 feature_1~590 (因為 PyCaret 對欄位名稱很敏感)
    print("⚠️ Warning: Training data not found. Cannot determine features.")
    return None

# --- Pytest Fixtures (測試前的準備工作) ---

@pytest.fixture(scope="module")
def model():
    path = get_model_path()
    if path is None:
        pytest.skip("❌ Model file not found. Skipping prediction tests (Expected in CI/CD without artifacts).")
    try:
        loaded_model = load_model(path)
        return loaded_model
    except Exception as e:
        pytest.fail(f"❌ Found model file but failed to load: {e}")

@pytest.fixture(scope="module")
def feature_columns():
    """取得正確的欄位名稱"""
    cols = get_required_features()
    if cols is None:
        pytest.skip("❌ Feature names not found. Cannot generate test data.")
    return cols

@pytest.fixture
def single_input_data(feature_columns):
    """產生一筆模擬的單點輸入資料 (全為 0)，但欄位名稱正確"""
    # 建立一個只有一列的 DataFrame，所有值設為 0
    data = pd.DataFrame(np.zeros((1, len(feature_columns))), columns=feature_columns)
    return data

@pytest.fixture
def batch_input_data(feature_columns):
    """產生一批模擬的輸入資料 (5 筆隨機數值)，欄位名稱正確"""
    np.random.seed(42)
    num_rows = 5
    # 產生隨機矩陣
    data_matrix = np.random.rand(num_rows, len(feature_columns))
    df = pd.DataFrame(data_matrix, columns=feature_columns)
    return df

# --- Test Cases (真正的測試邏輯) ---

def test_model_type(model):
    """測試 1: 確認模型物件載入成功"""
    assert model is not None, "Model failed to load"
    print("\n✅ Model loaded successfully")

def test_single_prediction_format(model, single_input_data):
    """測試 2: 單點預測功能與輸出格式"""
    try:
        # PyCaret 3.0 的 predict_model 會回傳包含原本 feature + prediction 的 DF
        predictions = predict_model(model, data=single_input_data)
        
        # 檢查關鍵輸出欄位是否存在
        cols = predictions.columns
        # PyCaret 3.x 通常輸出 'prediction_label' 和 'prediction_score'
        has_label = 'prediction_label' in cols
        has_score = 'prediction_score' in cols
        
        if not has_label:
            print(f"DEBUG: Output columns: {cols}")
        
        assert has_label, "Missing 'prediction_label' in output"
        assert has_score, "Missing 'prediction_score' in output"
        print("\n✅ Single prediction format check passed")
        
    except Exception as e:
        pytest.fail(f"Single prediction failed with error: {str(e)}")

def test_batch_prediction_shape(model, batch_input_data):
    """測試 3: 批次預測數量是否吻合"""
    try:
        predictions = predict_model(model, data=batch_input_data)
        assert len(predictions) == 5, f"Expected 5 predictions, got {len(predictions)}"
        print("\n✅ Batch prediction shape check passed")
    except Exception as e:
        pytest.fail(f"Batch prediction failed: {e}")

def test_prediction_values_range(model, batch_input_data):
    """測試 4: 預測機率是否合理 (0~1 之間)"""
    predictions = predict_model(model, data=batch_input_data)
    
    if 'prediction_score' in predictions.columns:
        scores = predictions['prediction_score']
        assert scores.min() >= 0.0, "Prediction score < 0 found"
        assert scores.max() <= 1.0, "Prediction score > 1 found"
        print("\n✅ Prediction score range check passed")