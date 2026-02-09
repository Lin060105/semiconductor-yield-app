# 使用輕量級的 Python 3.9 映像檔 (PyCaret 在 3.9 穩定性最佳)
FROM python:3.9-slim

# 設定工作目錄
WORKDIR /app

# 安裝系統依賴 (編譯某些 Python 套件需要 gcc)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 複製需求文件並安裝依賴
COPY requirements.txt .

# 升級 pip 並安裝套件 (增加 timeout 避免網路慢導致失敗)
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --timeout 100

# 複製專案所有檔案到容器內
COPY . .

# 暴露 Streamlit 預設埠口
EXPOSE 8501

# 設定啟動指令
# server.address=0.0.0.0 允許外部存取
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]