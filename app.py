import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import pickle
from pycaret.classification import load_model, predict_model

# --- 頁面配置 ---
st.set_page_config(page_title="半導體良率智慧診斷 V3.2", layout="wide")
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial'] 
plt.rcParams['axes.unicode_minus'] = False

# --- 載入資源 ---
@st.cache_resource
def get_resources():
    # 1. 載入模型
    pipeline = load_model('final_yield_prediction_model')
    model = pipeline.steps[-1][1] # 取出 Random Forest 模型
    
    # 2. 載入特徵清單 (取代原本讀取 CSV 的動作)
    with open('required_features.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    
    # 建立一個空的 DataFrame 作為 SHAP 的模板
    X_template = pd.DataFrame(columns=feature_names)
    
    # 3. 建立解釋器
    explainer = shap.TreeExplainer(model)
    
    return pipeline, explainer, feature_names

try:
    pipeline, explainer, feature_names = get_resources()
    model_loaded = True
except Exception as e:
    st.error(f"⚠️ 系統啟動失敗：{e}")
    st.info("請確認 'final_yield_prediction_model.pkl' 和 'required_features.pkl' 是否在目錄中。")
    model_loaded = False

# --- 工具函數 ---
@st.cache_data
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

# --- 側邊欄 ---
st.sidebar.title("🎛️ AI 中控台 (V3.2)")
app_mode = st.sidebar.radio("模式選擇", ["🔍 單筆診斷", "🚀 批量快篩"])
st.sidebar.markdown("---")

# 靈敏度調整
st.sidebar.header("⚖️ 判斷標準調整")
threshold = st.sidebar.slider("異常判定門檻 (Threshold)", 0.0, 1.0, 0.5, 0.05)

st.sidebar.markdown("---")
uploaded_file = st.sidebar.file_uploader("上傳晶圓數據 (CSV)", type="csv")

# 測試資料下載 (不需要依賴原始 CSV 了，這裡我們用假資料生成結構即可，或直接隱藏此功能)
# 為了演示方便，我們這裡改為「若有上傳檔案則顯示」

# --- 主畫面 ---
st.title("🏭 半導體良率智慧診斷系統")
st.caption("版本: V3.2 | 模型: Random Forest (SMOTE Enhanced) | 狀態: Ready")

if uploaded_file and model_loaded:
    try:
        df_in = pd.read_csv(uploaded_file)
        
        # 資料欄位檢查與對齊
        missing_cols = set(feature_names) - set(df_in.columns)
        if missing_cols:
            st.error(f"❌ 檔案格式錯誤！缺少以下欄位：{list(missing_cols)[:5]} ...等")
            st.stop()
            
        # 只保留需要的欄位
        df_process = df_in[feature_names]
        
        # 預測
        raw_predictions = predict_model(pipeline, data=df_process, raw_score=True)
        # 取得異常機率 (Label 1)
        probs = raw_predictions['prediction_score_1']
        final_labels = (probs >= threshold).astype(int)
        
        # --- 模式 1: 單筆診斷 ---
        if app_mode == "🔍 單筆診斷":
            idx = st.selectbox("選擇晶圓索引 (Index)", df_in.index)
            
            if st.button("進行診斷"):
                prob = probs[idx]
                is_fail = prob >= threshold
                
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.metric("異常機率 (Failure Probability)", f"{prob:.2%}")
                    if is_fail:
                        st.error("🔴 判定結果：FAIL (異常)")
                    else:
                        st.success("🟢 判定結果：PASS (良品)")
                
                with c2:
                    st.write("📊 關鍵特徵影響力 (SHAP)")
                    try:
                        # 處理 SHAP 維度
                        sv = explainer.shap_values(df_process.iloc[[idx]])
                        # 相容性處理 (針對不同版本的 SHAP/Sklearn)
                        if isinstance(sv, list): shap_val = sv[1][0]
                        elif len(sv.shape)==3: shap_val = sv[0][:,1]
                        else: shap_val = sv[0]
                        
                        fig, ax = plt.subplots()
                        shap.plots.waterfall(
                            shap.Explanation(shap_val, explainer.expected_value[1], 
                                           df_process.iloc[idx], feature_names),
                            show=False, max_display=10
                        )
                        st.pyplot(fig)
                    except Exception as e:
                        st.warning(f"SHAP 圖表繪製失敗: {e}")

        # --- 模式 2: 批量快篩 ---
        elif app_mode == "🚀 批量快篩":
            if st.button("執行全量檢測"):
                fails = df_in[final_labels == 1]
                
                st.subheader("檢測報告")
                m1, m2, m3 = st.columns(3)
                m1.metric("總檢測數量", len(df_in))
                m2.metric("🔴 預測異常數", len(fails))
                m3.metric("良率 (Yield)", f"{(1 - len(fails)/len(df_in)):.2%}")
                
                if not fails.empty:
                    st.warning("⚠️ 檢測到潛在異常晶圓 (依風險排序)")
                    # 建立結果表
                    res = fails.copy()
                    res['Risk_Score'] = probs[fails.index]
                    # 依照風險分數排序
                    res = res.sort_values('Risk_Score', ascending=False)
                    st.dataframe(res[['Risk_Score'] + feature_names[:5]].style.background_gradient(subset=['Risk_Score'], cmap='Reds'))
                else:
                    st.success("✅ 本批次未發現異常晶圓！")
                    
    except Exception as e:
        st.error(f"處理過程中發生錯誤: {e}")
else:
    if not model_loaded:
        st.warning("模型尚未載入，請檢查檔案。")
    else:
        st.info("👈 請從左側上傳 CSV 檔案開始分析")