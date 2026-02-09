# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2026-02-09
### 🚀 Features (新功能)
- **Advanced AutoML**: 整合 PyCaret (XGBoost, LightGBM, RF) 進行多模型競賽。
- **Interactive Dashboard**: 基於 Streamlit 的多頁籤介面，支援單筆與批次預測。
- **Fail Ranking**: 自動標記並排序高風險 (Fail) 產品。
- **Explainable AI**: 內建 SHAP Summary 與 Feature Importance 圖表。

### 🛠️ Infrastructure (基礎建設)
- **Dockerized**: 支援 Docker 一鍵部署 (`docker run`).
- **CI/CD**: GitHub Actions 自動執行 Linting 與 Pytest。
- **Testing**: 新增 `tests/test_predict.py` 確保預測管線穩定性。

### 📚 Documentation (文件)
- 新增 `docs/data_pipeline.md` 包含 Mermaid 架構圖。
- 更新 `README.md` 加入 Docker 使用教學。