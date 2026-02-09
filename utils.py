import streamlit as st
from pycaret.classification import load_model, predict_model
import os
import pickle
import pandas as pd

@st.cache_resource
def load_model_cached(model_path):
    """載入 PyCaret 模型並進行快取"""
    try:
        if not os.path.exists(f"{model_path}.pkl"):
            st.error(f"❌ 找不到模型檔案: {model_path}.pkl")
            return None
        return load_model(model_path)
    except Exception as e:
        st.error(f"❌ 無法載入模型: {e}")
        return None

def load_feature_config(config_path='required_features.pkl'):
    """載入特徵清單"""
    if os.path.exists(config_path):
        with open(config_path, 'rb') as f:
            return pickle.load(f)
    else:
        return ['feature_1', 'feature_2', 'feature_3']

def make_prediction(model, input_data):
    """單筆預測"""
    try:
        input_df = pd.DataFrame([input_data])
        predictions = predict_model(model, data=input_df)
        
        if 'prediction_label' in predictions.columns:
            pred_label = predictions['prediction_label'].iloc[0]
            pred_score = predictions['prediction_score'].iloc[0]
        else:
            pred_label = predictions['Label'].iloc[0]
            pred_score = predictions['Score'].iloc[0]
            
        return pred_label, pred_score
    except Exception as e:
        raise e

def make_batch_prediction(model, file):
    """
    執行批量預測
    Args:
        model: PyCaret 模型物件
        file: 上傳的 CSV 檔案物件
    Returns:
        pd.DataFrame: 包含原始資料與預測結果 (Label, Score)
    """
    try:
        # 1. 讀取上傳的 CSV
        data = pd.read_csv(file)
        
        # 2. 執行預測 (PyCaret 會自動處理缺失值與正規化)
        predictions = predict_model(model, data=data)
        
        # 3. 整理欄位名稱 (統一新舊版本 PyCaret 輸出)
        rename_dict = {
            'prediction_label': '預測結果 (Label)',
            'prediction_score': '信心分數 (Score)',
            'Label': '預測結果 (Label)',
            'Score': '信心分數 (Score)'
        }
        predictions = predictions.rename(columns=rename_dict)
        
        return predictions
    except Exception as e:
        raise Exception(f"批量預測失敗: {str(e)}")