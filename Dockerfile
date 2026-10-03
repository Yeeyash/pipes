# Container for the demo app: works on Hugging Face Spaces (Docker), Render, Fly.io, or any VM.
#   docker build -t weldvision .
#   docker run -p 8501:8501 weldvision      ->  http://localhost:8501
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Run as a non-root user with a writable home (required by Hugging Face Spaces).
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH \
    YOLO_CONFIG_DIR=/home/user/.config PORT=8501
WORKDIR /home/user/app
RUN mkdir -p /home/user/.config

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .
# Bake the NEU-DET test split into the image so the app starts without a download.
RUN python scripts/download_data.py --test-only

EXPOSE 8501
CMD ["sh", "-c", "exec streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0"]
