FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime@sha256:c8268a92a69bd500f8be0e665b2630ee006dadaf7bfbc24249141b15ff622755

LABEL org.opencontainers.image.source="https://github.com/anycloud-sh/perforated-resnet18-flowers102"
LABEL org.opencontainers.image.title="Perforated AI ResNet-18 Flowers-102 comparison"
LABEL org.opencontainers.image.licenses="Apache-2.0"

WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY runner.py .
COPY spot_checkpoint.py .
COPY upstream/ ./upstream/

ENV PYTHONUNBUFFERED=1
CMD ["python", "runner.py", "--epochs", "50"]
