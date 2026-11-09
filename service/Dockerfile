# Lab-to-Real detector service, CPU only. The image holds code only; the model bundle is mounted at run time.
#   docker build -t lab2real-detector:1.0 service
#   docker run --rm -p 8000:8000 -v "<repo>/model_bundles/lab2real-detector-v2:/model:ro" lab2real-detector:1.0
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    YOLO_CONFIG_DIR=/tmp/ultralytics MODEL_BUNDLE=/model YOLO_OFFLINE=1

WORKDIR /app
# CPU wheels only (no CUDA libraries), then the rest; OpenCV swapped for the headless build (no libGL needed)
RUN pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt /app/service/requirements.txt
RUN pip install -r /app/service/requirements.txt \
 && pip uninstall -y opencv-python \
 && pip install "opencv-python-headless<5" \
 && python -c "import cv2, torch, ultralytics; print(cv2.__version__, torch.__version__, ultralytics.__version__)"
COPY . /app/service

RUN useradd --create-home app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]
