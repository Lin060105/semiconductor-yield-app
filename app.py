import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from pycaret.classification import load_model, predict_model

# --- 頁面配置 ---
st.set_page_config(page_title="半導體良率智慧診斷 V3.0", layout="wide")
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] 
plt.rcParams['axes.unicode_minus'] = False

# --- 載入模型 ---
@st.cache_resource
def get_resources():
    # ✅ 正確寫法
    pipeline = load_model('final_yield_prediction_model')
    model = pipeline.steps[-1][1] # 取出 Random Forest 模型
    dataset = pd.read_csv('data/secom_processed.csv', nrows=5)
    X_template = dataset.drop('label', axis=1)
    explainer = shap.TreeExplainer(model)
    return pipeline, explainer, X_template

try:
    pipeline, explainer, X_template = get_resources()
    model_loaded = True
except Exception as e:
    st.error(f"模型載入失敗，詳細錯誤：{e}")
    # 為了除錯，我們多印一點資訊
    import os
    st.write("目前路徑下的檔案：", os.listdir('.'))
    model_loaded = False

# --- 工具函數 ---
@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

# --- 側邊欄：控制台 ---
st.sidebar.title("🎛️ AI 中控台")
app_mode = st.sidebar.radio("模式選擇", ["🔍 單筆診斷", "🚀 批量快篩"])

st.sidebar.markdown("---")
# 【Level 6 新功能】靈敏度調整
st.sidebar.header("⚖️ 判斷標準調整")
# 預設 0.5 (50%)，數值越低代表越嚴格 (只要有一點點像壞的就抓起來)
threshold = st.sidebar.slider("異常判定門檻 (Threshold)", 
                              min_value=0.0, max_value=1.0, value=0.5, step=0.05,
                              help="調低數值 = 篩網變密 (寧可錯殺)；調高數值 = 篩網變疏 (寧缺勿濫)")

st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("上傳晶圓數據 (CSV)", type="csv")

# 必殺範例下載
if st.sidebar.button("📥 下載含異常的測試檔"):
    df = pd.read_csv('secom_processed.csv')
    # 抓 5 個壞的，15 個好的
    fail = df[df['label']==1].sample(5, random_state=123)
    pass_ = df[df['label']==0].sample(15, random_state=123)
    final = pd.concat([fail, pass_]).sample(frac=1).drop('label', axis=1)
    st.sidebar.download_button("下載 CSV", convert_df(final), "test_data.csv", "text/csv")

# --- 主畫面 ---
st.title("🏭 半導體良率智慧診斷 V3.0")

if uploaded_file and model_loaded:
    df_in = pd.read_csv(uploaded_file)
    if 'label' in df_in.columns: df_in.drop('label', axis=1, inplace=True)
    
    # 預先計算所有預測機率 (這是關鍵！)
    # PyCaret 的 predict_model 預設只給 0/1，我們要用 raw_score=True 拿到機率
    raw_predictions = predict_model(pipeline, data=df_in, raw_score=True)
    # prediction_score_1 代表「是壞人(Fail)」的機率
    probs = raw_predictions['prediction_score_1']
    
    # 【核心邏輯】根據使用者設定的 Threshold 重新判斷
    # 如果 壞掉機率 > 門檻，就判為 1 (Fail)，否則 0 (Pass)
    final_labels = (probs >= threshold).astype(int)
    
    # 模式 1: 單筆
    if app_mode == "🔍 單筆診斷":
        idx = st.selectbox("選擇晶圓", df_in.index)
        if st.button("診斷"):
            prob = probs[idx]
            is_fail = prob >= threshold
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("異常機率 (AI Score)", f"{prob:.2%}")
                if is_fail:
                    st.error(f"🔴 判定：FAIL (異常) \n(因為機率 > {threshold})")
                else:
                    st.success(f"🟢 判定：PASS (良品) \n(因為機率 < {threshold})")
            
            with col2:
                # SHAP
                try:
                    # 簡單處理維度問題
                    sv = explainer.shap_values(df_in.iloc[[idx]])
                    if isinstance(sv, list): sv = sv[1][0]
                    elif len(sv.shape)==3: sv = sv[0][:,1]
                    else: sv = sv[0]
                    
                    fig, ax = plt.subplots()
                    shap.plots.waterfall(shap.Explanation(sv, explainer.expected_value[1], df_in.iloc[idx], df_in.columns), show=False)
                    st.pyplot(fig)
                except: st.warning("無法繪製 SHAP 圖")

    # 模式 2: 批量
    elif app_mode == "🚀 批量快篩":
        if st.button("執行快篩"):
            fails = df_in[final_labels == 1]
            passes = df_in[final_labels == 0]
            
            st.metric("目前門檻值", threshold)
            c1, c2, c3 = st.columns(3)
            c1.metric("總數", len(df_in))
            c2.metric("🔴 偵測異常", len(fails))
            c3.metric("🟢 判定良品", len(passes))
            
            if not fails.empty:
                st.error("⚠️ 異常清單 (根據當前門檻篩選):")
                # 顯示機率讓使用者參考
                res_df = fails.copy()
                res_df['異常機率'] = probs[fails.index]
                st.dataframe(res_df.style.highlight_max(axis=0))
            else:
                st.success("🎉 本批次無異常 (可嘗試調低門檻值)")

else:

    st.info("👈 請上傳數據並調整門檻值來測試 AI 行為。")


