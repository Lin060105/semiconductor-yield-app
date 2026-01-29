# 🏭 Semiconductor Yield Prediction System | 半導體良率智慧診斷系統

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](請把這裡改成您的Streamlit網址)
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![PyCaret](https://img.shields.io/badge/ML-PyCaret-yellow.svg)](https://pycaret.org/)

> **這是一個基於 AI 的半導體產線輔助系統，能夠透過感測器數據預測晶圓是否異常，並提供可解釋性分析 (SHAP) 與動態決策門檻調整功能。**

---

## 🚀 線上展示 (Live Demo)

👉 **[點擊這裡開啟 AI 診斷系統]((https://semiconductor-yield-app-tmyu9jwd7kii2zndseugtq.streamlit.app/))**

*(建議使用電腦瀏覽器開啟以獲得最佳體驗)*

---

## 💡 專案亮點 (Key Features)

這個專案解決了傳統半導體檢測依賴人工複檢、效率低落的問題。

### 1. 🔍 單筆深度診斷 (Single Prediction)
- 針對單一晶圓數據進行即時分析。
- **可解釋性 AI (XAI)**：整合 **SHAP Waterfall Plot**，視覺化呈現導致異常的關鍵特徵（例如：`feature_492` 數值過高），讓工程師知道「為什麼壞掉」。

### 2. 🚀 批量快速篩選 (Batch Processing)
- 支援上傳 CSV 檔案進行整批晶圓快篩。
- 自動標記高風險晶圓，大幅縮短檢測時間。

### 3. ⚖️ 動態靈敏度調整 (Dynamic Threshold)
- **業界實戰思維**：內建「決策門檻拉桿」，允許工程師根據產線需求調整 AI 的嚴格程度。
- **Trade-off**：想要「寧可錯殺，不可放過」或是「減少誤判」？由使用者決定。

---

## 🛠️ 技術棧 (Tech Stack)

- **核心語言**：Python 3.10
- **機器學習**：AutoML (PyCaret), Random Forest, Scikit-learn
- **資料處理**：Pandas, NumPy
- **視覺化與介面**：Streamlit, Matplotlib, SHAP
- **雲端部署**：Streamlit Cloud

---


## 💻 如何在本地端執行 (Installation)

如果您想在自己的電腦上運行此專案：

1. **Clone 專案**
```bash
git clone https://github.com/Lin060105/semiconductor-yield-app.git
```

2. **安裝依賴套件**
```bash
pip install -r requirements.txt
```

3. **啟動系統**
```bash
streamlit run app.py
```

## 📝 開發紀錄

- **Level 1**: 數據清洗與 UCI 資料集特徵工程。
- **Level 2**: 使用 PyCaret 比較 10+ 種演算法，選定 Random Forest。
- **Level 3**: 解決資料不平衡問題，優化 Recall 率。
- **Level 6**: 加入商業邏輯（Threshold Tuning），解決模型過於保守的問題。
- **Level 7**: 成功部署至 Streamlit Cloud。
