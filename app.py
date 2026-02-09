import streamlit as st
from PIL import Image
import os
import pandas as pd
import utils

# --- 1. 設定頁面 ---
st.set_page_config(page_title="半導體良率預測系統", page_icon="🏭", layout="wide")
st.title("🏭 半導體良率預測 App (v2.1)")
st.markdown("### 智慧製造良率分析平台 | SHAP Explainable AI")

# --- 2. 載入資源 ---
model = utils.load_model_cached('final_yield_prediction_model')
required_features = utils.load_feature_config()

# --- 3. 側邊欄 ---
st.sidebar.image("https://img.icons8.com/color/96/000000/chip.png", width=80)
st.sidebar.title("功能選單")
menu = st.sidebar.radio("", ["單筆預測", "批量預測 (Batch)", "模型效能報告"])

# --- 功能 A: 單筆預測 ---
if menu == "單筆預測":
    st.subheader("🔍 單筆資料即時檢測")
    with st.form("prediction_form"):
        col_input = st.columns(3)
        input_data = {}
        for i, feature in enumerate(required_features[:6]):
            with col_input[i % 3]:
                input_data[feature] = st.number_input(f"{feature}", value=0.0, format="%.4f")
        
        if len(required_features) > 6:
             for feature in required_features[6:]:
                 input_data[feature] = 0.0
        submit = st.form_submit_button("🚀 開始分析")

    if submit and model:
        try:
            label, score = utils.make_prediction(model, input_data)
            st.divider()
            c1, c2 = st.columns(2)
            with c1:
                if label == 1:
                    st.error("🛑 預測結果：Fail (不良品)")
                else:
                    st.success("✅ 預測結果：Pass (良品)")
            with c2:
                st.metric("AI 信心分數", f"{score:.2%}")
        except Exception as e:
            st.error(f"預測錯誤: {e}")

# --- 功能 B: 批量預測 (升級版!) ---
elif menu == "批量預測 (Batch)":
    st.subheader("📂 批量資料上傳檢測")
    uploaded_file = st.file_uploader("上傳 CSV 檔案", type=["csv"])
    
    if uploaded_file is not None and model:
        if st.button("🚀 開始批量分析"):
            with st.spinner("正在進行 AI 推論與風險排序..."):
                try:
                    result_df = utils.make_batch_prediction(model, uploaded_file)
                    
                    # 統計
                    fail_df = result_df[result_df['預測結果 (Label)'] == 1]
                    fail_count = len(fail_df)
                    total_count = len(result_df)
                    fail_rate = fail_count / total_count
                    
                    # 顯示 KPI
                    m1, m2, m3 = st.columns(3)
                    m1.metric("總檢測數", f"{total_count} 顆")
                    m2.metric("預測不良品數", f"{fail_count} 顆", delta_color="inverse")
                    m3.metric("預測不良率", f"{fail_rate:.1%}", delta_color="inverse")
                    
                    st.divider()
                    
                    # --- 新功能: 高風險排名 ---
                    st.subheader("🏆 高風險不良品 TOP 10 (Fail Ranking)")
                    st.info("以下是模型認為「最像不良品」的前 10 筆資料，建議優先檢查。")
                    
                    if fail_count > 0:
                        # 依照信心分數降序排列 (假設分數越高代表越像 Label 1)
                        # 注意：PyCaret 的 Score 針對預測的 Label。如果是 Label 1，Score 越高越危險。
                        # 如果是 Label 0，Score 越高越安全。
                        # 這裡我們只取預測為 1 (Fail) 的資料來排序
                        
                        top_fails = fail_df.sort_values(by='信心分數 (Score)', ascending=False).head(10)
                        
                        # 顯示時稍微美化一下，把重要的欄位往前放
                        cols = ['預測結果 (Label)', '信心分數 (Score)'] + [c for c in top_fails.columns if c not in ['預測結果 (Label)', '信心分數 (Score)']]
                        st.dataframe(top_fails[cols].style.background_gradient(subset=['信心分數 (Score)'], cmap='Reds'))
                    else:
                        st.success("🎉 太棒了！本次檢測未發現不良品。")

                    # 下載區
                    st.divider()
                    st.subheader("📥 下載報告")
                    csv = result_df.to_csv(index=False).encode('utf-8')
                    st.download_button("下載完整 CSV", csv, "yield_prediction_results.csv", "text/csv")
                    
                except Exception as e:
                    st.error(f"分析失敗: {e}")

# --- 功能 C: 報告 (新增 SHAP) ---
elif menu == "模型效能報告":
    st.subheader("📊 模型訓練報告")
    
    report_images = {
        "SHAP AI 解釋 (新!)": "SHAP Summary.png",
        "特徵重要性": "Feature Importance.png",
        "混淆矩陣": "Confusion Matrix.png",
        "ROC 曲線": "AUC.png"
    }
    
    tabs = st.tabs(list(report_images.keys()))
    
    for i, (title, filename) in enumerate(report_images.items()):
        with tabs[i]:
            path = os.path.join("reports", filename)
            
            # 特別為 SHAP 頁面加一些說明
            if "SHAP" in title:
                st.markdown("""
                **如何閱讀這張圖？**
                * **Y軸 (左邊)**：特徵名稱，越上面的特徵對良率影響越大。
                * **顏色 (紅/藍)**：紅色代表數值高，藍色代表數值低。
                * **X軸 (下方)**：對模型的影響。往**右**代表傾向預測為 **Fail (1)**，往**左**代表傾向 **Pass (0)**。
                * *例如：如果某特徵呈現「紅色在右邊」，表示該數值越高，越容易導致產品壞掉。*
                """)
            
            if os.path.exists(path):
                st.image(Image.open(path), caption=title, use_container_width=True)
            else:
                if "SHAP" in title:
                    st.warning("⚠️ 尚未生成 SHAP 圖表。請執行新的 `train_upgrade.py`。")
                else:
                    st.warning(f"⚠️ 找不到報告: {filename}")

st.markdown("---")
st.caption("Powered by Lin060105 | Semiconductor Yield App v2.1")