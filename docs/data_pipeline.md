# 🏭 Data Pipeline & Model Training Workflow

本文檔詳細說明半導體良率預測系統的資料處理流與模型訓練架構。

## 🛠️ 系統架構流程圖 (Mermaid)

```mermaid
graph TD
    %% 定義樣式
    classDef data fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef script fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    classDef artifact fill:#fff3e0,stroke:#ef6c00,stroke-width:2px;

    %% 節點定義
    RawData[("📂 Raw Data<br/>(secom.data / labels)")]:::data
    ScriptPre[("🐍 scripts/01_data_preprocessing.py<br/>(資料清洗腳本)")]:::script
    
    ProcessedData[("📄 secom_processed.csv<br/>(已清洗資料)")]:::data
    
    ScriptTrain[("🐍 train_upgrade.py<br/>(模型訓練與升級腳本)")]:::script
    
    subgraph AutoML[PyCaret AutoML Engine]
        Setup[環境設定<br/>(Fix Imbalance / Normalize)]
        Compare[模型競賽<br/>(RF vs XGBoost vs LightGBM)]
        Tune[最佳模型優化]
    end
    
    Model[("🤖 final_yield_prediction_model.pkl<br/>(最終模型)")]:::artifact
    Reports[("📊 Evaluation Reports<br/>(SHAP, AUC, Confusion Matrix)")]:::artifact
    
    App[("🚀 Streamlit App<br/>(app.py)")]:::script

    %% 流程連線
    RawData --> ScriptPre
    ScriptPre -->|去除常量, 填補缺失值| ProcessedData
    
    ProcessedData --> ScriptTrain
    ScriptTrain --> Setup
    Setup --> Compare
    Compare -->|選出 AUC 最高者| Tune
    
    Tune --> Model
    Tune --> Reports
    
    Model --> App
    Reports --> App

    ---

### 第二部分：資料處理細節

這部分說明了前面的清洗邏輯。

```markdown
## 📊 資料處理細節 (Data Preprocessing)

### 1. 資料清洗 (`scripts/01_data_preprocessing.py`)
原始 SECOM 數據集包含大量缺失值 (NaN) 與冗餘特徵，我們執行以下處理：
* **缺失值處理**：使用 KNN Imputer 或 Mean/Median 填補。
* **特徵篩選**：
    * 移除單一值 (Constant) 欄位。
    * 移除高相關性 (High Correlation) 特徵以避免共線性。
* **格式統一**：合併 Feature 與 Label，輸出為標準 CSV 格式。


---

### 第三部分：模型訓練與輸出產物

這部分說明了 AutoML 機制和最終產出的檔案。

### 2. 模型訓練與評估 (`train_upgrade.py`)
使用 **PyCaret** 框架進行自動化機器學習：
* **不平衡處理 (Fix Imbalance)**：由於良率資料通常 Pass 遠多於 Fail，我們使用 SMOTE 或類似技術平衡樣本。
* **多模型比較**：同時訓練 Random Forest, XGBoost, LightGBM，依據 **AUC** 指標自動選擇最佳模型。
* **可解釋性 AI (XAI)**：
    * 整合 **SHAP (SHapley Additive exPlanations)** 計算特徵貢獻度。
    * 生成 Confusion Matrix 確認召回率 (Recall)。

## 📁 輸出產物
執行訓練後，系統會生成以下關鍵檔案供 App 使用：
1.  `final_yield_prediction_model.pkl`: 封裝好的預測管線。
2.  `reports/SHAP Summary.png`: 全局特徵影響力分析圖。
3.  `reports/model_comparison.csv`: 各模型效能評比表。