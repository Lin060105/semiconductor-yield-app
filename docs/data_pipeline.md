# 🏭 資料處理與模型訓練流程 (Data Pipeline)

本文檔詳細說明半導體良率預測系統的資料流向、前處理邏輯與模型訓練策略。

## 📊 1. 系統架構圖 (Pipeline Overview)

```mermaid
graph TD
    A[原始資料 SECOM Dataset] -->|讀取| B(資料前處理 Data Preprocessing)
    B --> C{特徵工程 Feature Engineering}
    C -->|清洗後資料| D[儲存: secom_processed.csv]
    D --> E[PyCaret AutoML 環境]
    
    subgraph "AutoML 訓練階段"
        E --> F[類別平衡 SMOTE]
        F --> G[模型比較與選擇]
        G --> H[Random Forest 模型]
        H --> I[模型最佳化 Tuned Model]
    end
    
    I --> J[最終模型 Final Model]
    J --> K[部署: Streamlit App]
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style J fill:#bbf,stroke:#333,stroke-width:2px
    style K fill:#bfb,stroke:#333,stroke-width:2px

    ---

### 第 2 部分：資料前處理細節

```markdown
## 🛠️ 2. 資料前處理細節 (Preprocessing Steps)

原始資料包含 1567 筆樣本與 591 個感測器特徵，且存在大量缺失值與標記不平衡問題。

### 步驟 A: 資料清洗 (Cleaning)
- **去除無效特徵**：刪除缺失值超過 55% 的欄位 (Drop features with >55% missing values)。
- **單一值移除**：刪除只有單一數值的欄位 (Drop columns with only 1 unique value)，因其不具預測力。
- **缺失值填補**：使用平均值 (Mean imputation) 填補剩餘缺失數據。

### 步驟 B: 特徵篩選 (Feature Selection)
- 利用 `Variance Threshold` 移除低變異特徵。
- 透過相關性矩陣 (Correlation Matrix) 移除高度共線性特徵。

## ⚙️ 3. 模型訓練策略 (Training Strategy)

### 類別不平衡處理 (Imbalance Handling)
由於良品 (Pass) 遠多於不良品 (Fail)，比例約為 14:1。直接訓練會導致模型傾向預測全為良品。
- **解決方案**：在 PyCaret setup 中啟用 `fix_imbalance=True`。
- **演算法**：使用 **SMOTE (Synthetic Minority Over-sampling Technique)** 合成少數類樣本，使訓練集達到平衡。

### 模型選擇 (Model Selection)
- **演算法**：Random Forest Classifier (隨機森林)。
- **選擇原因**：
    1. 對雜訊與離群值具有良好的魯棒性 (Robustness)。
    2. 內建特徵重要性評估 (Feature Importance)。
    3. 不容易過度擬合 (Overfitting)。

    
## 📈 4. 評估指標 (Evaluation Metrics)

本專案不僅關注 Accuracy，更重視對不良品 (Fail, Label=1) 的捕捉能力：

1.  **Recall (召回率)**：最關鍵指標。我們寧可誤判良品為壞品 (False Alarm)，也不能放過真正的壞品 (Miss)。
2.  **AUC (Area Under Curve)**：評估模型整體的鑑別力。
3.  **Confusion Matrix**：觀察 TP (真正壞品) 與 FN (漏網之魚) 的具體數量。

## 🔄 5. 自動化報告 (Automated Reporting)
訓練腳本 (`train_upgrade.py`) 會自動生成以下圖表至 `reports/` 目錄：
- `Confusion Matrix.png`: 混淆矩陣
- `Feature Importance.png`: 關鍵影響因子
- `SHAP Summary.png`: 模型可解釋性分析 (Explainable AI)