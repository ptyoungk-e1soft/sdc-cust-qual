#!/bin/bash
# 디스플레이 결함 분석 시스템 데모 실행 스크립트

echo "============================================"
echo "  디스플레이 결함 분석 시스템 - Demo"
echo "  Cosmos Reason VLM + GraphRAG"
echo "============================================"

# 작업 디렉토리 이동
cd "$(dirname "$0")/.."

# GPU 사용 가능 여부 확인
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "[GPU 정보]"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
fi

# Docker 환경에서 실행
echo "Docker 컨테이너로 데모 시작..."
sudo docker run --rm -it \
    --gpus all \
    -v "$(pwd)":/workspace \
    -w /workspace \
    -p 7860:7860 \
    nvcr.io/nvidia/pytorch:24.12-py3 \
    bash -c "
        pip install -q gradio transformers==4.49.0 peft==0.15.0 qwen-vl-utils accelerate 2>&1 | tail -2
        echo ''
        echo '============================================'
        echo '  Demo Server Starting...'
        echo '  URL: http://localhost:7860'
        echo '============================================'
        echo ''
        python demo/app.py
    "
