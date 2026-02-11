# 🏭 半導體良率預測 - 資料處理與建模流程

本文件詳細說明專案的資料流向 (Data Pipeline)，從原始 SECOM 數據集到最終的模型部署。

## 🛠️ 數據處理流程圖 (Pipeline Visualization)

```mermaid
graph TD
    %% 定義樣式
    classDef storage fill:#f9f,stroke:#333,stroke-width:2px;
    classDef process fill:#e1f5fe,stroke:#0277bd,stroke-width:2px;
    classDef model fill:#fff9c4,stroke:#fbc02d,stroke-width:2px;
    classDef output fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;

    %% 節點定義
    RawData[("📂 原始數據 (SECOM)")]:::storage
    Cleaning["🧹 資料清洗\n(去除常數特徵, 填補遺失值)"]:::process
    FeatureEng["⚙️ 特徵工程\n(相關性過濾, 降維)"]:::process
    ProcessedData[("💾 處理後數據\n(secom_processed.csv)")]:::storage
    
    subgraph PyCaret Training [PyCaret 自動化訓練環境]
        Setup["⚖️ 環境設定 (Setup)\n(SMOTE 不平衡處理, 正規化)"]:::process
        Compare["🏎️ 模型競賽\n(RF, XGBoost, LightGBM, CatBoost)"]:::model
        BestModel["🏆 選定最佳模型\n(CatBoost Classifier)"]:::model
        Tuning["🔧 模型優化與校準"]:::model
    end

    Eval["📊 模型評估\n(AUC, Recall, Confusion Matrix)"]:::output
    Explain["🧠 模型解釋\n(SHAP Values Analysis)"]:::output
    Deployment["🚀 Streamlit App 部署"]:::output

    %% 連線與流程
    RawData --> Cleaning
    Cleaning --> FeatureEng
    FeatureEng --> ProcessedData
    ProcessedData --> Setup
    Setup --> Compare
    Compare --> BestModel
    BestModel --> Tuning
    Tuning --> Eval
    Eval --> Explain
    BestModel --> Deployment


## 📝 詳細步驟說明

### 1. 資料前處理 (Data Preprocessing)
* **來源**：UCI SECOM Dataset (1567 筆樣本, 591 個感測器特徵)。
* **清洗**：
    * 剔除缺失值超過 50% 的欄位。
    * 移除單一數值（變異數為 0）的無效特徵。
    * 使用中位數 (Median) 填補剩餘缺失值。
* **輸出**：生成 `secom_processed.csv`，保留約 400+ 個有效特徵。

### 2. 模型訓練 (Model Training)
使用 **PyCaret** 框架進行自動化機器學習：
* **不平衡處理**：由於良率異常 (Fail) 樣本極少 (~6%)，訓練過程使用 **SMOTE** 進行過採樣。
* **模型比較**：針對 Recall (召回率) 進行優化，比較了 Random Forest, XGBoost, LightGBM 與 CatBoost。
* **最終選擇**：**CatBoost** 因在 Recall 與 F1-Score 表現最佳而被選為最終模型。


### 3. 可解釋性分析 (Explainability)
為了讓工程師理解預測依據，整合了 **SHAP (SHapley Additive exPlanations)**：
* 計算每個感測器對良率判定的貢獻度。
* 生成 Summary Plot 以視覺化特徵重要性。