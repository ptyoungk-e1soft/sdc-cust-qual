# Cosmos Reason VLM 기반 디스플레이 불량 분석 시스템
FROM nvcr.io/nvidia/pytorch:24.01-py3

WORKDIR /app

# 시스템 의존성
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

# 소스 코드 복사
COPY configs/ configs/
COPY src/ src/
COPY scripts/ scripts/

# 데이터 디렉토리 생성
RUN mkdir -p data/raw data/processed data/annotations data/sft output logs

# 환경 변수
ENV PYTHONPATH=/app
ENV CONFIG_PATH=/app/configs/inference.yaml
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface
ENV HF_HOME=/root/.cache/huggingface

# 포트 노출
EXPOSE 8000

# 헬스체크
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# 기본 명령 (API 서버)
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
