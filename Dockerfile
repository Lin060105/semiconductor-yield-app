# 使用輕量級的 Python 3.9 環境 (PyCaret 3.x 支援度最佳)
FROM python:3.9-slim

# 設定工作目錄
WORKDIR /app

# 1. 安裝系統層級依賴
# build-essential: 編譯某些 Python 套件需要
# libgomp1: LightGBM 與 CatBoost 運算需要的 OpenMP 函式庫
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 2. 複製與安裝 Python 套件
COPY requirements.txt .
# --no-cache-dir 減小映像檔體積
RUN pip install --no-cache-dir -r requirements.txt

# 3. 複製專案所有程式碼
COPY . .

# 4. 暴露 Streamlit 預設 Port
EXPOSE 8501

# 5. 健康檢查 (Optional, 增加生產環境穩定性)
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# 6. 啟動指令
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]