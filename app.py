import streamlit as st
import pandas as pd
import numpy as np
from pycaret.classification import load_model, predict_model
import shap
import matplotlib.pyplot as plt
import os

# --- 設定頁面資訊 ---
st.set_page_config(
    page_title="Semiconductor Yield Prediction",
    page_icon="🧊",
    layout="wide"
)

# --- 標題與簡介 ---
st.title("🧊 AI Semiconductor Yield Prediction System")
st.markdown("""
**Status**: v1.0.0 (Production Ready) | **Model**: CatBoost/XGBoost Ensemble
This application predicts wafer yield outcomes and analyzes failure root causes using SHAP.
""")

# --- 側邊欄：模型與設定 ---
st.sidebar.header("🔧 Configuration")
model_path = 'output/final_yield_prediction_model'

@st.cache_resource
def load_yield_model():
    if os.path.exists(model_path + '.pkl'):
        return load_model(model_path)
    else:
        st.error(f"Model file not found at {model_path}.pkl. Please run training scripts first.")
        return None

pipeline = load_yield_model()

if pipeline:
    st.sidebar.success("Model Loaded Successfully")
    try:
        model = pipeline._final_estimator
    except:
        model = pipeline

# --- 主功能分頁 ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📂 Batch Prediction", 
    "📊 Batch Statistics", 
    "⚠️ Fail Ranking", 
    "🔍 SHAP Analysis",
    "📉 Model Performance" 
])

# 初始化 session_state
if 'predictions' not in st.session_state:
    st.session_state['predictions'] = None
if 'data' not in st.session_state:
    st.session_state['data'] = None

# ==========================================
# Tab 1: Batch Prediction
# ==========================================
with tab1:
    st.subheader("Upload Wafer Data for Prediction")
    use_sample = st.checkbox("Use sample data (secom_processed.csv)")
    uploaded_file = st.file_uploader("Or upload your CSV file", type=['csv'])
    
    df = None
    if use_sample:
        if os.path.exists('data/secom_processed.csv'):
            df = pd.read_csv('data/secom_processed.csv').head(100)
            st.info("Loaded sample data (first 100 rows).")
    elif uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.info("File uploaded successfully.")

    if df is not None:
        st.session_state['data'] = df
        if st.button("🚀 Run Prediction", type="primary"):
            with st.spinner("Analyzing wafers..."):
                predictions = predict_model(pipeline, data=df)
                st.session_state['predictions'] = predictions
                st.success("Prediction complete! Check other tabs for insights.")
        
        with st.expander("Preview Raw Data"):
            st.dataframe(df.head())

        # 🌟 新增：讓使用者可以一鍵下載完整的預測結果
        if st.session_state['predictions'] is not None:
            csv_all = st.session_state['predictions'].to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 下載完整預測結果 (CSV)",
                data=csv_all,
                file_name="full_predictions_result.csv",
                mime="text/csv"
            )

# ==========================================
# Tab 2: Batch Statistics
# ==========================================
with tab2:
    st.subheader("Batch Yield Overview")
    if st.session_state['predictions'] is not None:
        preds = st.session_state['predictions']
        total = len(preds)
        fail_count = preds[preds['prediction_label'] == 1].shape[0]
        pass_count = total - fail_count
        yield_rate = (pass_count / total) * 100
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Wafers", f"{total}")
        col2.metric("Yield Rate", f"{yield_rate:.2f}%", delta_color="normal")
        col3.metric("Defect Count", f"{fail_count}", delta_color="inverse")
        
        fig, ax = plt.subplots()
        ax.pie([pass_count, fail_count], labels=['Pass', 'Fail'], autopct='%1.1f%%', colors=['#66b3ff','#ff9999'])
        st.pyplot(fig)
    else:
        st.warning("Please run prediction in 'Batch Prediction' tab first.")

# ==========================================
# Tab 3: Fail Ranking
# ==========================================
with tab3:
    st.subheader("Top High-Risk Wafers")
    if st.session_state['predictions'] is not None:
        preds = st.session_state['predictions']
        fails = preds[preds['prediction_label'] == 1].copy()
        
        if not fails.empty:
            top_fails = fails.sort_values(by='prediction_score', ascending=False).head(20)
            st.dataframe(top_fails.style.background_gradient(subset=['prediction_score'], cmap='Reds'))
            
            # 🌟 新增：明顯的專屬下載按鈕 (針對高風險晶圓)
            csv_fails = top_fails.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="🚨 下載 Top 20 高風險晶圓清單 (CSV)",
                data=csv_fails,
                file_name="high_risk_wafers.csv",
                mime="text/csv",
                type="primary" # 使用 primary 顏色讓按鈕更醒目
            )
        else:
            st.success("No failures predicted in this batch!")
            st.markdown("---")
            st.markdown("**Lowest Confidence 'Pass' Wafers (Potential False Negatives):**")
            risky_pass = preds[preds['prediction_label'] == 0].sort_values(by='prediction_score', ascending=True).head(10)
            st.dataframe(risky_pass)
            
            # 🌟 新增：即使沒有 Fail，也可以下載潛在風險清單
            csv_risky = risky_pass.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 下載潛在風險晶圓清單 (CSV)",
                data=csv_risky,
                file_name="risky_pass_wafers.csv",
                mime="text/csv"
            )
    else:
        st.warning("Please run prediction first.")

# ==========================================
# Tab 4: SHAP Analysis 
# ==========================================
with tab4:
    st.subheader("Model Interpretability (SHAP)")
    
    st.markdown("### 🌟 全局特徵重要性 (Global Summary)")
    st.success("**SHAP Summary 就是我們這一步最核心的成果圖！這張圖是用來回答「模型為什麼會這樣預測？」的關鍵證據。**")
    
    # 這裡會精準抓取你 reports 資料夾下的 SHAP Summary.png
    shap_img_path = "reports/SHAP Summary.png"
    
    if os.path.exists(shap_img_path):
        st.image(shap_img_path, caption="SHAP Summary Plot", use_container_width=True)
    else:
        st.info(f"尚未找到 SHAP 圖片，請確認 `{shap_img_path}` 檔案是否存在。")

    st.markdown("---")
    st.markdown("### 🔍 單筆晶圓深度分析 (Local Waterfall)")
    
    if st.session_state['data'] is not None:
        shap_data = st.session_state['data'].head(500)
        try:
            transformer = pipeline[:-1]
            X_transformed = transformer.transform(shap_data)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_transformed)
            
            # 🌟 修復核心：處理不同模型 (XGBoost/CatBoost) 產生的 SHAP 格式差異
            if isinstance(shap_values, list):
                # 如果是 list，代表有 Class 0 和 Class 1 兩個陣列，我們取 Class 1 (Fail)
                sv = shap_values[1]
                bv = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
            elif len(np.array(shap_values).shape) == 3: # 處理 3D 陣列的罕見情況
                sv = np.array(shap_values)[:, :, 1]
                bv = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
            else:
                # 單純陣列格式
                sv = shap_values
                bv = explainer.expected_value
                if isinstance(bv, (list, np.ndarray)) and len(bv) == 1:
                    bv = bv[0]
            
            sample_idx = st.selectbox("Select Wafer Index for Deep Dive", shap_data.index)
            loc_idx = shap_data.index.get_loc(sample_idx)
            
            # 精準抓取單筆資料建立 Explanation 物件，避免全域切片產生的 IndexError
            row_data = X_transformed.iloc[loc_idx] if isinstance(X_transformed, pd.DataFrame) else X_transformed[loc_idx]
            
            explanation = shap.Explanation(
                values=sv[loc_idx], 
                base_values=bv, 
                data=row_data, 
                feature_names=X_transformed.columns if hasattr(X_transformed, 'columns') else None
            )
            
            st.markdown(f"**Why Wafer {sample_idx} is predicted this way:**")
            fig_water, ax_water = plt.subplots()
            shap.plots.waterfall(explanation, show=False)
            st.pyplot(fig_water)
            
        except Exception as e:
            st.error(f"Could not generate dynamic SHAP plot: {e}")
    else:
        st.warning("Please load data first in the 'Batch Prediction' tab.")

# ==========================================
# Tab 5: Model Performance 
# ==========================================
with tab5:
    st.subheader("📊 Model Validation & Performance Proof")
    st.markdown("Detailed metrics demonstrating model reliability and robustness.")
    
    # 這裡只留下正常的模型評估圖表，不會再有找不到圖片的報錯了
    report_imgs = {
        "Confusion Matrix": "output/automl_reports/confusion_matrix.png",
        "AUC-ROC Curve": "output/automl_reports/auc_roc_curve.png",
        "Feature Importance": "output/automl_reports/feature_importance.png",
        "Learning Curve (Overfitting Check)": "output/automl_reports/learning_curve.png",
        "Model Comparison (XGB vs CatBoost)": "reports/model_comparison_final.png"
    }

    col1, col2 = st.columns(2)
    
    for i, (title, path) in enumerate(report_imgs.items()):
        if os.path.exists(path):
            with (col1 if i % 2 == 0 else col2):
                st.image(path, caption=title, use_container_width=True)
        else:
            with (col1 if i % 2 == 0 else col2):
                st.warning(f"Image not found: {title}")
    
    st.markdown("---")
    st.subheader("📝 Overfitting Analysis Report")
    analysis_path = "reports/overfitting_analysis.txt"
    if os.path.exists(analysis_path):
        with open(analysis_path, "r", encoding='utf-8') as f:
            report_text = f.read()
        st.text_area("Analysis Result", report_text, height=150)
    else:
        st.info("Analysis text report not found.")