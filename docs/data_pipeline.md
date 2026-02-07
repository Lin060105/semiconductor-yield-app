# 🛠️ Data Pipeline & Model Architecture

本專案採用 PyCaret 自動化機器學習框架進行開發，針對半導體製造數據（SECOM Dataset）的特性，設計了完整的資料處理流程。

## 1. 資料預處理 (Preprocessing)

原始數據包含 1567 筆晶圓製程記錄，每筆記錄擁有 591 個感測器特徵（Sensors）。

### 數據清洗
- **缺失值處理 (Imputation)**：
  - 數值型特徵：使用 `Mean`（平均值）填補。
  - 類別型特徵：使用 `Mode`（眾數）填補。
- **特徵篩選 (Feature Selection)**：
  - 移除高共線性（Multicollinearity）特徵，閾值設為 0.9。
  - 移除零變異數（Zero Variance）特徵。
  - 最終保留特徵數：**474 個**。

## 2. 不平衡資料處理 (Handling Class Imbalance)

### 問題背景
原始資料集存在極度的類別不平衡：
- **Pass (良品, 0)**: 1463 筆 (93.4%)
- **Fail (瑕疵, 1)**: 104 筆 (6.6%)

若直接訓練，模型傾向於預測所有晶圓為「良品」以獲得高準確率（Accuracy），但這會導致漏檢（Recall = 0），這在半導體產業是不可接受的。

### 解決方案：SMOTE
我們使用了 **SMOTE (Synthetic Minority Over-sampling Technique)** 技術：
- **原理**：在少數類別（Fail）樣本之間進行插值，合成新的人造樣本，而非單純複製。
- **效果**：將訓練集中的 Fail 樣本數提升，使模型能學習到異常晶圓的特徵邊界。
- **PyCaret 設定**：`fix_imbalance=True`。

## 3. 模型選擇與優化

經過多模型比較（XGBoost, LightGBM, Random Forest 等），最終選用 **Random Forest (隨機森林)**：
- **選擇理由**：
  1. 對高維度數據（High-dimensional data）魯棒性強。
  2. 能夠提供特徵重要性（Feature Importance），便於工程師追蹤製程問題。
  3. 配合 SMOTE 後，在 Recall (瑕疵檢出率) 上表現最佳。