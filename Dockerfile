# Google Cloud Run 用 Dockerfile
# Streamlit + OpenCV + MediaPipe アプリケーション

FROM python:3.10-slim

# 環境変数設定
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# OpenCV/MediaPipe 用のシステム依存をインストール
# ffmpeg を追加して H.264 (avc1) コーデック対応
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリを設定
WORKDIR /workspace

# requirements.txt を先にコピーして pip install（Dockerキャッシュ最適化）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコードをコピー
COPY . .

# Streamlit のアドレス設定
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# ポートを公開（Cloud Run の PORT 環境変数を使用）
EXPOSE 8080

# Streamlit アプリを起動（Cloud Run の PORT 環境変数を使用）
CMD streamlit run app.py --server.port=${PORT:-8080} --server.address=0.0.0.0

