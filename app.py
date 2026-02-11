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

# --- 主功能分頁 (新增 Tab 5: Model Performance) ---
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
        else:
            st.success("No failures predicted in this batch!")
            st.markdown("---")
            st.markdown("**Lowest Confidence 'Pass' Wafers (Potential False Negatives):**")
            risky_pass = preds[preds['prediction_label'] == 0].sort_values(by='prediction_score', ascending=True).head(10)
            st.dataframe(risky_pass)
    else:
        st.warning("Please run prediction first.")

# ==========================================
# Tab 4: SHAP Analysis
# ==========================================
with tab4:
    st.subheader("Model Interpretability (SHAP)")
    if st.session_state['data'] is not None:
        analysis_type = st.radio("Select Analysis Type", ["Global Summary (Feature Importance)", "Local Waterfall (Single Wafer)"])
        shap_data = st.session_state['data'].head(500)
        
        try:
            transformer = pipeline[:-1]
            X_transformed = transformer.transform(shap_data)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_transformed)
            
            if analysis_type == "Global Summary (Feature Importance)":
                fig_shap, ax_shap = plt.subplots()
                shap.summary_plot(shap_values, X_transformed, show=False)
                st.pyplot(fig_shap)
            elif analysis_type == "Local Waterfall (Single Wafer)":
                sample_idx = st.selectbox("Select Wafer Index", shap_data.index)
                loc_idx = shap_data.index.get_loc(sample_idx)
                
                explanation = shap.Explanation(
                    values=shap_values, 
                    base_values=explainer.expected_value, 
                    data=X_transformed, 
                    feature_names=X_transformed.columns if hasattr(X_transformed, 'columns') else None
                )
                
                st.markdown(f"**Why Wafer {sample_idx} is predicted this way:**")
                fig_water, ax_water = plt.subplots()
                shap.plots.waterfall(explanation[loc_idx], show=False)
                st.pyplot(fig_water)
        except Exception as e:
            st.error(f"Could not generate SHAP plot: {e}")
    else:
        st.warning("Please load data first.")

# ==========================================
# Tab 5: Model Performance (模型證明)
# ==========================================
with tab5:
    st.subheader("📊 Model Validation & Performance Proof")
    st.markdown("Detailed metrics demonstrating model reliability and robustness.")
    
    # 定義圖片路徑
    report_imgs = {
        "Confusion Matrix": "output/automl_reports/confusion_matrix.png",
        "AUC-ROC Curve": "output/automl_reports/auc_roc_curve.png",
        "Feature Importance": "output/automl_reports/feature_importance.png",
        "Learning Curve (Overfitting Check)": "output/automl_reports/learning_curve.png",
        "Model Comparison (XGB vs CatBoost)": "reports/model_comparison_final.png"
    }

    # 使用 2 欄佈局顯示圖片
    col1, col2 = st.columns(2)
    
    # 遍歷並顯示圖片
    for i, (title, path) in enumerate(report_imgs.items()):
        # 檢查檔案是否存在
        if os.path.exists(path):
            # 輪流放在左欄或右欄
            with (col1 if i % 2 == 0 else col2):
                st.image(path, caption=title, use_container_width=True)
        else:
            with (col1 if i % 2 == 0 else col2):
                st.warning(f"Image not found: {title} ({path})")
    
    # 顯示文字版的過擬合分析
    st.markdown("---")
    st.subheader("📝 Overfitting Analysis Report")
    analysis_path = "reports/overfitting_analysis.txt"
    if os.path.exists(analysis_path):
        with open(analysis_path, "r") as f:
            report_text = f.read()
        st.text_area("Analysis Result", report_text, height=150)
    else:
        st.info("Analysis text report not found.")