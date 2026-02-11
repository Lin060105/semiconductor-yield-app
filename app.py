import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from pycaret.classification import load_model, predict_model
import matplotlib.pyplot as plt

# --- 1. 頁面設定 ---
st.set_page_config(
    page_title="Semiconductor Yield Prediction Pro",
    page_icon="🏭",
    layout="wide"
)

st.title("🏭 半導體良率預測系統 (v2.0 Pro)")
st.markdown("""
本系統利用 **CatBoost / Random Forest** 整合模型預測晶片良率 (Pass/Fail)。
並提供模型解釋 (SHAP) 與多模型效能比較報告。
""")

# --- 2. 載入模型與資源 ---
@st.cache_resource
def load_prediction_model():
    # 優先載入新的最佳模型，若無則載入舊的
    if os.path.exists('final_yield_prediction_model.pkl'):
        return load_model('final_yield_prediction_model')
    elif os.path.exists('reports/final_yield_prediction_model.pkl'):
        return load_model('reports/final_yield_prediction_model')
    else:
        st.error("❌ 找不到模型檔案，請先執行 train_upgrade.py")
        return None

model = load_prediction_model()

# 載入特徵清單 (確保輸入順序正確)
try:
    if os.path.exists('required_features.pkl'):
        with open('required_features.pkl', 'rb') as f:
            required_features = pickle.load(f)
    else:
        st.warning("⚠️ 找不到 required_features.pkl，將使用預設特徵。")
        required_features = [f'Sensor_{i}' for i in range(1, 11)]
except Exception as e:
    st.error(f"⚠️ 載入特徵清單失敗: {e}")
    required_features = [f'Sensor_{i}' for i in range(1, 11)]

# --- 3. 建立分頁 ---
tab1, tab2, tab3 = st.tabs(["🔍 單點/批次預測", "📊 模型解釋 (SHAP)", "🏆 模型效能報告"])

# === Tab 1: 預測功能 ===
with tab1:
    st.header("線上預測與模擬")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("輸入感測器數值")
        input_data = {}
        # 為了演示，只顯示前 5 個特徵的輸入框
        display_features = required_features[:5] if len(required_features) > 5 else required_features
        for feature in display_features:
            val = st.number_input(f"{feature}", value=0.0)
            input_data[feature] = val
        
        if len(required_features) > 5:
            st.caption(f"*(已隱藏剩餘 {len(required_features)-5} 個特徵，預設為 0)*")
            # 其他特徵補 0 (模擬)
            for feature in required_features[5:]:
                input_data[feature] = 0.0
            
        predict_btn = st.button("🚀 執行預測", type="primary")

    with col2:
        st.subheader("預測結果")
        if predict_btn and model:
            try:
                df_input = pd.DataFrame([input_data])
                prediction = predict_model(model, data=df_input)
                
                # PyCaret 3.x 輸出欄位處理
                label_col = 'prediction_label' if 'prediction_label' in prediction.columns else 'Label'
                score_col = 'prediction_score' if 'prediction_score' in prediction.columns else 'Score'
                
                if label_col in prediction.columns:
                    result = prediction[label_col].iloc[0]
                    score = prediction[score_col].iloc[0]
                    
                    # 假設 1 或 '1' 為 Fail
                    if str(result) == '1' or result == 1: 
                        st.error(f"⚠️ 預測結果: **Fail (異常)**")
                        st.metric("異常機率 (Confidence)", f"{score:.2%}")
                        st.warning("建議行動：檢查 Sensor 數值是否偏離製程規範。")
                    else:
                        st.success(f"✅ 預測結果: **Pass (正常)**")
                        st.metric("信心水準", f"{score:.2%}")
                else:
                    st.error("無法解析預測結果，欄位名稱不符。")
                    st.write(prediction.columns)
            except Exception as e:
                st.error(f"預測執行錯誤: {e}")

        st.markdown("---")
        st.subheader("📂 批次上傳預測")
        uploaded_file = st.file_uploader("上傳 CSV 檔案 (需包含所有感測器欄位)", type="csv")
        if uploaded_file and model:
            try:
                batch_df = pd.read_csv(uploaded_file)
                # 檢查關鍵欄位是否存在
                missing_cols = [col for col in required_features if col not in batch_df.columns]
                
                if not missing_cols:
                    predictions = predict_model(model, data=batch_df)
                    st.success("✅ 批次預測完成！")
                    st.write(predictions.head())
                    
                    # 下載結果
                    csv = predictions.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 下載預測結果", csv, "predictions.csv", "text/csv")
                else:
                    st.error(f"❌ 檔案缺少以下欄位: {missing_cols[:3]}...")
            except Exception as e:
                st.error(f"檔案讀取失敗: {e}")

# === Tab 2: 模型解釋 (SHAP) ===
with tab2:
    st.header("🧠 模型解釋：為什麼會 Fail？")
    st.info("此頁面展示 SHAP (SHapley Additive exPlanations) 分析，幫助工程師理解哪些感測器數值對良率影響最大。")
    
    # 顯示靜態生成的 SHAP 圖
    shap_img_path = os.path.join("reports", "SHAP Summary.png")
    if os.path.exists(shap_img_path):
        st.image(shap_img_path, caption="全域特徵重要性 (Global Feature Importance)", use_column_width=True)
    else:
        st.warning("⚠️ 尚未生成 SHAP Summary 圖表。請確認 train_upgrade.py 已完整執行。")
        
    st.markdown("### 💡 如何解讀？")
    st.markdown("""
    - **特徵排序**：由上而下代表影響力由大到小。
    - **顏色**：紅色代表數值較高，藍色代表數值較低。
    - **SHAP Value**：向右偏代表增加 Fail 機率，向左偏代表增加 Pass 機率。
    """)

# === Tab 3: 模型效能報告 ===
with tab3:
    st.header("🏆 多模型評估報告")
    
    # 1. 顯示比較表格
    csv_path = os.path.join("reports", "model_comparison.csv")
    if os.path.exists(csv_path):
        st.subheader("模型指標排行榜")
        df_metrics = pd.read_csv(csv_path)
        st.dataframe(df_metrics.style.highlight_max(axis=0, subset=['AUC', 'Recall', 'F1'], color='lightgreen'))
        st.caption("註：Recall (召回率) 對於偵測半導體失效最為重要。")
    else:
        st.warning("⚠️ 尚未找到 model_comparison.csv，請先執行 train_upgrade.py")

    # 2. 顯示圖表 Gallery
    st.subheader("📊 詳細圖表")
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("**混淆矩陣 (Confusion Matrix)**")
        cm_path = os.path.join("reports", "Confusion Matrix.png")
        if os.path.exists(cm_path):
            st.image(cm_path, use_column_width=True)
        else:
            st.info("*(圖表未生成)*")

        st.markdown("**PR 曲線 (Precision-Recall)**")
        pr_path = os.path.join("reports", "Precision Recall.png")
        if os.path.exists(pr_path):
            st.image(pr_path, use_column_width=True)
        else:
             st.info("*(圖表未生成)*")

    with col_b:
        st.markdown("**ROC 曲線 (AUC)**")
        auc_path = os.path.join("reports", "AUC.png")
        if os.path.exists(auc_path):
            st.image(auc_path, use_column_width=True)
        else:
            st.info("*(圖表未生成)*")
            
        st.markdown("**學習曲線 (Learning Curve)**")
        lc_path = os.path.join("reports", "Learning Curve.png")
        if os.path.exists(lc_path):
            st.image(lc_path, use_column_width=True)
        else:
             st.info("ℹ️ 學習曲線未生成 (可能已跳過或運算中)")

st.sidebar.info(f"當前使用模型: {model.__class__.__name__ if model else '未載入'}")