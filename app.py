import streamlit as st
import pandas as pd
import numpy as np
import os
import shutil
from pycaret.classification import load_model, predict_model
from PIL import Image

# --- 1. 頁面設定 ---
st.set_page_config(
    page_title="Semiconductor Yield Prediction Pro",
    page_icon="🏭",
    layout="wide"
)

st.title("🏭 半導體良率預測系統 (v3.0 Ultimate)")
st.markdown("""
本系統利用 **CatBoost / Random Forest** 整合模型預測晶片良率。
新功能：**Fail Ranking** (優先處理高風險晶片) 與 **完整模型評估報告**。
""")

# --- 2. 載入模型與資源 ---
@st.cache_resource
def load_prediction_model():
    # 嘗試多個路徑載入模型
    paths = [
        'output/final_yield_prediction_model', 
        'final_yield_prediction_model',
        'reports/final_yield_prediction_model'
    ]
    
    for path in paths:
        # PyCaret load_model 不需要 .pkl 副檔名
        if os.path.exists(path + '.pkl'):
            try:
                return load_model(path)
            except:
                continue
    return None

model = load_prediction_model()
if not model:
    st.error("❌ 找不到模型檔案，請確認 `output/final_yield_prediction_model.pkl` 存在。")

# 載入特徵清單 (確保輸入順序正確)
required_features = [f'Sensor_{i}' for i in range(1, 11)] # 預設 fallback
try:
    if os.path.exists('required_features.pkl'):
        import pickle
        with open('required_features.pkl', 'rb') as f:
            required_features = pickle.load(f)
except Exception as e:
    pass # 使用預設值

# --- 3. 建立分頁 ---
tab1, tab2, tab3 = st.tabs(["🔍 預測與高風險清單", "📊 模型解釋 (SHAP)", "🏆 模型效能報告"])

# === Tab 1: 預測功能 (含 Fail Ranking) ===
with tab1:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("單點模擬")
        input_data = {}
        # 為了演示，只顯示前 5 個特徵
        display_features = required_features[:5]
        for feature in display_features:
            val = st.number_input(f"{feature}", value=0.0)
            input_data[feature] = val
            
        predict_btn = st.button("🚀 執行模擬預測", type="primary")

        if predict_btn and model:
            # 補齊其他特徵為 0
            for feature in required_features[5:]:
                input_data[feature] = 0.0
                
            df_input = pd.DataFrame([input_data])
            prediction = predict_model(model, data=df_input)
            
            # 處理 PyCaret 3.x 輸出
            try:
                label = prediction['prediction_label'].iloc[0]
                score = prediction['prediction_score'].iloc[0]
                
                if label == 1:
                    st.error(f"⚠️ 預測結果: **Fail (異常)**")
                    st.metric("異常機率", f"{score:.2%}")
                else:
                    st.success(f"✅ 預測結果: **Pass (正常)**")
                    st.metric("安全信心", f"{score:.2%}")
            except Exception as e:
                st.error(f"解析錯誤: {e}")

    with col2:
        st.subheader("📂 批次預測 & Fail Ranking")
        st.info("上傳 CSV 檔案，系統將自動篩選出 **高風險 (High Probability of Fail)** 的晶片。")
        
        uploaded_file = st.file_uploader("上傳測試數據 (CSV)", type="csv")
        if uploaded_file and model:
            try:
                batch_df = pd.read_csv(uploaded_file)
                predictions = predict_model(model, data=batch_df)
                
                # 確保欄位名稱一致
                lbl_col = 'prediction_label'
                score_col = 'prediction_score'
                
                if lbl_col in predictions.columns:
                    # --- 關鍵功能：Fail Ranking ---
                    st.markdown("### 🔥 高風險晶片排行榜 (Top Failures)")
                    
                    # 篩選預測為 Fail (1) 的資料
                    fail_df = predictions[predictions[lbl_col] == 1].copy()
                    
                    if not fail_df.empty:
                        # 依照分數排序 (分數越高代表越像 Fail)
                        fail_df = fail_df.sort_values(by=score_col, ascending=False)
                        
                        # 顯示前 10 名
                        st.dataframe(
                            fail_df.head(10).style.background_gradient(subset=[score_col], cmap='Reds'),
                            use_container_width=True
                        )
                        st.warning(f"⚠️ 共發現 {len(fail_df)} 個潛在異常晶片！建議優先檢查上表中的項目。")
                    else:
                        st.success("🎉 太棒了！本批次數據中沒有發現預測為 Fail 的晶片。")
                    
                    # 下載完整結果
                    csv = predictions.to_csv(index=False).encode('utf-8')
                    st.download_button("📥 下載完整預測報告", csv, "predictions_with_ranking.csv", "text/csv")
                else:
                    st.error("預測結果欄位不如預期，無法生成排行榜。")
            except Exception as e:
                st.error(f"批次處理失敗: {e}")

# === Tab 2: 模型解釋 (SHAP) ===
with tab2:
    st.header("🧠 SHAP 模型解釋")
    # 支援多個可能的圖片路徑
    shap_paths = [
        'output/automl_reports/shap_summary_plot.png', # Step 1 可能產生的路徑
        'reports/SHAP Summary.png', 
        'output/shap_plots/shap_summary_plot.png'
    ]
    
    img_found = False
    for p in shap_paths:
        if os.path.exists(p):
            st.image(p, caption="Feature Importance (SHAP)", use_column_width=True)
            img_found = True
            break
            
    if not img_found:
        st.warning("⚠️ 尚未生成 SHAP 圖表。請執行 `scripts/05_explain_model.py` 或確認路徑。")

# === Tab 3: 模型效能報告 (整合 Step 1 結果) ===
with tab3:
    st.header("🏆 模型效能儀表板")
    
    # 1. 比較表格
    # Step 1 生成的是 'model_comparison_benchmark.csv'
    csv_path = 'reports/model_comparison_benchmark.csv'
    if os.path.exists(csv_path):
        st.subheader("模型基準測試 (Benchmark)")
        df_metrics = pd.read_csv(csv_path)
        # 簡單清理表格
        if 'Unnamed: 0' in df_metrics.columns:
            df_metrics = df_metrics.drop(columns=['Unnamed: 0'])
        st.dataframe(df_metrics.style.highlight_max(axis=0, color='lightgreen'))
    else:
        st.info("ℹ️ 尚未找到模型比較表 (model_comparison_benchmark.csv)。")

    # 2. 圖表展示
    st.subheader("📊 視覺化評估")
    
    # 定義圖表路徑 (根據 Step 1 的輸出設定)
    # Step 1 存到 output/automl_reports/
    img_dir = 'output/automl_reports' 
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("**學習曲線 (Learning Curve) - 過擬合檢查**")
        p = os.path.join(img_dir, 'learning_curve.png')
        if os.path.exists(p):
            st.image(p, use_column_width=True)
        else:
            st.info("(尚無學習曲線圖)")
            
        st.markdown("**混淆矩陣 (Confusion Matrix)**")
        p = os.path.join(img_dir, 'confusion_matrix.png')
        if os.path.exists(p):
            st.image(p, use_column_width=True)
        else:
            st.info("(尚無混淆矩陣圖)")

    with col_b:
        st.markdown("**AUC 曲線 (ROC Curve)**")
        p = os.path.join(img_dir, 'auc_roc_curve.png')
        if os.path.exists(p):
            st.image(p, use_column_width=True)
        else:
            st.info("(尚無 AUC 圖)")

        st.markdown("**特徵重要性 (Feature Importance)**")
        p = os.path.join(img_dir, 'feature_importance.png')
        if os.path.exists(p):
            st.image(p, use_column_width=True)
        else:
            st.info("(尚無特徵重要性圖)")