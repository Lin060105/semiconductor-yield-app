# 🏭 Semiconductor Yield Prediction System (v3.0 Ultimate)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://semiconductor-yield-app-tmyu9jwd7kii2zndseugtq.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-Passing-success?style=for-the-badge)](tests/)

> **A professional AI system for semiconductor yield diagnosis, featuring Fail Ranking, SHAP Explainability, and Business-Driven Threshold Tuning.**
>
> **基於 AI 的半導體產線智慧診斷系統，整合「高風險晶片排序」、「可解釋性分析」與「動態決策門檻」。**

---

## 🚀 Live Demo (線上展示)

👉 **[Click Here to Launch App (點擊開啟 AI 診斷系統)](https://semiconductor-yield-app-mw4jsvcuklcgwpcnqmy7gq.streamlit.app/)**

---

## 💡 Key Features (專案亮點)

### 1. 🔥 Fail Ranking System (高風險排序) **[NEW]**
- **Pain Point**: Traditional methods require reviewing thousands of records.
- **Solution**: Our system automatically filters and ranks wafers with the highest probability of failure (Score > 0.5), allowing engineers to prioritize the "Top 10 Riskiest Chips" instantly.

### 2. 🧠 Explainable AI (SHAP 分析)
- **Why it failed?**: Visualizes root causes using **SHAP Summary Plots**.
- Identifies critical sensors (e.g., `Sensor_59` drift) contributing to yield loss, moving beyond "Black Box" predictions.

### 3. ⚖️ Business-Driven Threshold (商業決策調整)
- Includes a dynamic slider to adjust the classification threshold.
- Allows balancing between **Overkill (False Positive)** and **Escapes (False Negative)** based on current market costs.

---

## 💰 Business Context: Cost Matrix Analysis

In semiconductor manufacturing, not all errors cost the same. We optimized the model based on the following reality:

| Actual \ Predicted | Predicted Pass (0) | Predicted Fail (1) |
| :--- | :--- | :--- |
| **Actual Pass (0)** | ✅ **True Negative**<br>Normal Shipment<br>(Cost: $0) | ⚠️ **False Positive**<br>Re-test Cost / Scrap Good Die<br>(Cost: Low) |
| **Actual Fail (1)** | ❌ **False Negative**<br>Client Return / Reputation Loss<br>(Cost: **Very High**) | ✅ **True Positive**<br>Defect Interception<br>(Cost: Saved!) |

**Strategy**: Our model prioritizes **Recall** to minimize "False Negatives" (preventing bad chips from reaching customers).

---

## 📂 Project Structure

```text
├── .github/workflows/   # CI/CD Pipeline (GitHub Actions)
├── data/                # SECOM Dataset
├── output/              # Trained Models & Plots
├── reports/             # Performance Metrics (CSV) & Learning Curves
├── scripts/             # Core Logic (Preprocessing, Training, Eval)
├── tests/               # Automated Tests (Pytest)
├── app.py               # Streamlit Application
├── Dockerfile           # Container Configuration
└── README.md            # Documentation
```

---

## 🛠️ Tech Stack & MLOps

- **Core**: Python 3.9, Pandas, NumPy
- **Modeling**: PyCaret, Random Forest, CatBoost, Scikit-learn (SMOTE)
- **Explainability**: SHAP (SHapley Additive exPlanations)
- **DevOps**: Docker, GitHub Actions (CI/CD), Streamlit Cloud
- **Quality Assurance**: Pytest (Automated Unit Testing)

---

## 💻 Installation & Usage

### Method 1: Local Development

**1. Clone the repository**
```bash
git clone https://github.com/Lin060105/semiconductor-yield-app.git
cd semiconductor-yield-app
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the App**
```bash
streamlit run app.py
```

### Method 2: Docker Deployment

Deploy anywhere with a consistent environment.

**Build Image**
```bash
docker build -t yield-app .
```

**Run Container**
```bash
docker run -p 8501:8501 yield-app
```

---

## 📈 Model Performance (Benchmark)

We compared multiple algorithms to ensure optimal performance:

| Model | AUC | Recall | Status |
| :--- | :--- | :--- | :--- |
| **Random Forest** | 0.78 | High | ✅ Selected (Best Stability) |
| XGBoost | 0.76 | Medium | Benchmark |
| LightGBM | 0.75 | Medium | Benchmark |

*(See `reports/model_comparison.csv` for full details.)*

---

## 📝 Development Log (里程碑)

| Level | Milestone | Status |
| :--- | :--- | :--- |
| Lv 1 | Data Cleaning & Feature Engineering | ✅ Done |
| Lv 2 | Algorithm Comparison (PyCaret) | ✅ Done |
| Lv 3 | Handling Imbalance (SMOTE) | ✅ Done |
| Lv 6 | Business Logic (Threshold Tuning) | ✅ Done |
| Lv 7 | Streamlit Cloud Deployment | ✅ Done |
| Lv 8 | Dockerization & CI/CD Pipeline | ✅ Completed (v3.0) |
| Lv 9 | Fail Ranking & Automated Reporting | ✅ Completed (v3.0) |

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.
