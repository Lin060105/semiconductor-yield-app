import streamlit as st
import pandas as pd
import numpy as np
from pycaret.classification import load_model, predict_model
import shap
import matplotlib.pyplot as plt
import os

# --- 1. 設定頁面資訊 (移除側邊欄後，Layout 更重要) ---
st.set_page_config(
    page_title="Semiconductor Yield Prediction",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="collapsed" # 預設收起側邊欄
)

# --- 2. 載入模型 (邏輯移出側邊欄) ---
# 設定模型路徑
model_path = 'output/final_yield_prediction_model'

@st.cache_resource
def load_yield_model():
    """載入模型並回傳 Pipeline"""
    if os.path.exists(model_path + '.pkl'):
        return load_model(model_path)
    else:
        return None

# 在主流程中載入模型
with st.spinner("Loading AI Model and Resources..."):
    pipeline = load_yield_model()

# 檢查模型是否載入成功，並設定 model 變數
if pipeline is None:
    st.error(f"❌ Critical Error: Model file not found at '{model_path}.pkl'. Please run training scripts first.")
    st.stop() # 停止執行後續程式碼
else:
    # 嘗試提取最終模型供 SHAP 使用
    try:
        model = pipeline._final_estimator
    except:
        model = pipeline

# --- 3. 標題與簡介 (整合狀態顯示) ---
st.title("🧊 AI Semiconductor Yield Prediction System")

# 使用 Columns 來讓狀態顯示更緊湊
col_desc, col_status = st.columns([3, 1])
with col_desc:
    st.markdown("""
    **Overview**: This application predicts wafer yield outcomes and analyzes failure root causes using SHAP values.
    Upload your batch data to identify high-risk wafers immediately.
    """)
with col_status:
    # 用一個漂亮的綠色區塊顯示狀態，取代原本的側邊欄
    st.success("✅ System Status: Online\n\nModel: CatBoost/XGBoost Ensemble")

st.markdown("---")

# --- 4. 主功能分頁 (UI 英文統一) ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📂 Batch Prediction", 
    "📊 Statistics", 
    "⚠️ Fail Ranking", 
    "🔍 Root Cause (SHAP)",
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
    st.subheader("Data Upload & Execution")
    
    col_input, col_action = st.columns([2, 1])
    
    with col_input:
        use_sample = st.checkbox("Use Sample Data (secom_processed.csv)")
        uploaded_file = st.file_uploader("Or Upload CSV File", type=['csv'])
    
    df = None
    if use_sample:
        if os.path.exists('data/secom_processed.csv'):
            df = pd.read_csv('data/secom_processed.csv').head(100)
            st.info("ℹ️ Loaded sample data (first 100 rows).")
    elif uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully.")

    if df is not None:
        st.session_state['data'] = df
        
        # 把按鈕放在右側 Action 區塊，比較整齊
        with col_action:
            st.write("###") #用來對齊的空白
            if st.button("🚀 Run Prediction", type="primary", use_container_width=True):
                with st.spinner("Processing wafers..."):
                    predictions = predict_model(pipeline, data=df)
                    st.session_state['predictions'] = predictions
                    st.success("Analysis Complete!")
        
        with st.expander("👁️ Preview Input Data"):
            st.dataframe(df.head())

        # 下載按鈕區域
        if st.session_state['predictions'] is not None:
            st.divider()
            st.subheader("Downloads")
            csv_all = st.session_state['predictions'].to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 Download Full Results (CSV)",
                data=csv_all,
                file_name="full_predictions_result.csv",
                mime="text/csv"
            )

# ==========================================
# Tab 2: Batch Statistics
# ==========================================
with tab2:
    st.subheader("Yield Overview")
    if st.session_state['predictions'] is not None:
        preds = st.session_state['predictions']
        total = len(preds)
        fail_count = preds[preds['prediction_label'] == 1].shape[0]
        pass_count = total - fail_count
        yield_rate = (pass_count / total) * 100
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Wafers", f"{total}")
        c2.metric("Yield Rate", f"{yield_rate:.2f}%")
        c3.metric("Defect Count", f"{fail_count}", delta_color="inverse")
        
        # 讓圖表置中且不要太大
        col_fig, _ = st.columns([1, 1])
        with col_fig:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.pie([pass_count, fail_count], labels=['Pass', 'Fail'], autopct='%1.1f%%', colors=['#66b3ff','#ff9999'])
            st.pyplot(fig)
    else:
        st.warning("⚠️ Please run prediction in the 'Batch Prediction' tab first.")

# ==========================================
# Tab 3: Fail Ranking
# ==========================================
with tab3:
    st.subheader("High-Risk Wafer Ranking")
    if st.session_state['predictions'] is not None:
        preds = st.session_state['predictions']
        fails = preds[preds['prediction_label'] == 1].copy()
        
        if not fails.empty:
            st.markdown("**Top 20 Wafers with Highest Failure Probability:**")
            top_fails = fails.sort_values(by='prediction_score', ascending=False).head(20)
            st.dataframe(top_fails.style.background_gradient(subset=['prediction_score'], cmap='Reds'))
            
            csv_fails = top_fails.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="🚨 Download Top 20 High-Risk List (CSV)",
                data=csv_fails,
                file_name="high_risk_wafers.csv",
                mime="text/csv",
                type="primary"
            )
        else:
            st.success("🎉 No failures predicted in this batch!")
            st.divider()
            st.markdown("**Lowest Confidence 'Pass' Wafers (Watch List):**")
            risky_pass = preds[preds['prediction_label'] == 0].sort_values(by='prediction_score', ascending=True).head(10)
            st.dataframe(risky_pass)
            
            csv_risky = risky_pass.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 Download Watch List (CSV)",
                data=csv_risky,
                file_name="risky_pass_wafers.csv",
                mime="text/csv"
            )
    else:
        st.warning("⚠️ Please run prediction first.")

# ==========================================
# Tab 4: SHAP Analysis 
# ==========================================
with tab4:
    st.subheader("Model Interpretability")
    
    st.markdown("### 1. Global Feature Importance")
    st.caption("Visualizes which sensor readings contribute most to yield failures across the entire dataset.")
    
    shap_img_path = "reports/SHAP Summary.png"
    
    if os.path.exists(shap_img_path):
        st.image(shap_img_path, caption="SHAP Summary Plot", use_container_width=True)
    else:
        st.info(f"SHAP Summary image not found at `{shap_img_path}`.")

    st.divider()
    st.markdown("### 2. Local Waterfall Analysis")
    st.caption("Deep dive into a specific wafer to understand why the model predicted it as Fail/Pass.")
    
    if st.session_state['data'] is not None:
        shap_data = st.session_state['data'].head(500) # Limit for performance
        
        # 選擇晶圓 ID
        col_sel, col_viz = st.columns([1, 3])
        
        with col_sel:
            sample_idx = st.selectbox("Select Wafer Index:", shap_data.index)
            
        with col_viz:
            try:
                # 準備 SHAP 資料
                transformer = pipeline[:-1]
                X_transformed = transformer.transform(shap_data)
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_transformed)
                
                # 處理 SHAP 格式 (相容 XGBoost/CatBoost)
                if isinstance(shap_values, list):
                    sv = shap_values[1]
                    bv = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
                elif len(np.array(shap_values).shape) == 3:
                    sv = np.array(shap_values)[:, :, 1]
                    bv = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
                else:
                    sv = shap_values
                    bv = explainer.expected_value
                    if isinstance(bv, (list, np.ndarray)) and len(bv) == 1:
                        bv = bv[0]
                
                loc_idx = shap_data.index.get_loc(sample_idx)
                row_data = X_transformed.iloc[loc_idx] if isinstance(X_transformed, pd.DataFrame) else X_transformed[loc_idx]
                
                explanation = shap.Explanation(
                    values=sv[loc_idx], 
                    base_values=bv, 
                    data=row_data, 
                    feature_names=X_transformed.columns if hasattr(X_transformed, 'columns') else None
                )
                
                st.markdown(f"**Impact Factors for Wafer {sample_idx}:**")
                fig_water, ax_water = plt.subplots()
                shap.plots.waterfall(explanation, show=False)
                st.pyplot(fig_water)
                
            except Exception as e:
                st.error(f"Error generating SHAP plot: {e}")
    else:
        st.warning("⚠️ Please load data first in the 'Batch Prediction' tab.")

# ==========================================
# Tab 5: Model Performance 
# ==========================================
with tab5:
    st.subheader("Validation Metrics")
    st.markdown("Detailed proof of model reliability.")
    
    report_imgs = {
        "Confusion Matrix": "output/automl_reports/confusion_matrix.png",
        "AUC-ROC Curve": "output/automl_reports/auc_roc_curve.png",
        "Feature Importance": "output/automl_reports/feature_importance.png",
        "Learning Curve": "output/automl_reports/learning_curve.png",
        "Model Comparison": "reports/model_comparison_final.png"
    }

    col1, col2 = st.columns(2)
    
    for i, (title, path) in enumerate(report_imgs.items()):
        container = col1 if i % 2 == 0 else col2
        with container:
            if os.path.exists(path):
                st.image(path, caption=title, use_container_width=True)
            else:
                st.warning(f"⚠️ Missing: {title}")
    
    st.divider()
    st.subheader("Overfitting Analysis")
    analysis_path = "reports/overfitting_analysis.txt"
    if os.path.exists(analysis_path):
        with open(analysis_path, "r", encoding='utf-8') as f:
            report_text = f.read()
        st.text_area("Analysis Report", report_text, height=150)
    else:
        st.info("No analysis report found.")
