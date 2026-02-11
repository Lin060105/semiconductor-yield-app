# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-02-11
### Added
- **Docker Support**: Added `Dockerfile` for containerized deployment.
- **CI/CD**: Added `tests/` directory with `pytest` for automated testing.
- **Model Upgrade**: Integrated **CatBoost** classifier for better recall on imbalanced data.
- **Explainability**: Added **SHAP Summary Plot** tab in Streamlit app.
- **Documentation**: Added `docs/data_pipeline.md` with Mermaid diagram.
- **Metrics**: Added `reports/model_comparison.csv` to track RF, XGBoost, and CatBoost performance.

### Changed
- **UI Overhaul**: Updated `app.py` to use Streamlit Tabs layout (Prediction, Explainability, Performance).
- **Dependency Management**: Locked versions in `requirements.txt` to prevent PyCaret/Sklearn conflicts.
- **README**: Rewritten `README.md` with Docker instructions, Tech Stack, and Business Context.

### Fixed
- Fixed class imbalance issue using **SMOTE** in training pipeline.
- Resolved scikit-learn version compatibility issues with PyCaret 3.0.

## [0.1.0] - 2025-10-15
### Added
- Initial release of Semiconductor Yield Prediction App.
- Basic Random Forest model implementation.
- Simple Streamlit interface for single prediction.
- Data preprocessing script for SECOM dataset.