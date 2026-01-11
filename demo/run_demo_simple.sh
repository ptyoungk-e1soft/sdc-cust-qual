#!/bin/bash
# 간단한 데모 실행 (GPU 없이 시뮬레이션 모드)

echo "============================================"
echo "  디스플레이 결함 분석 시스템 - Demo"
echo "  (시뮬레이션 모드 - GPU 불필요)"
echo "============================================"

cd "$(dirname "$0")/.."

# 가상환경 활성화 (있으면)
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# 필요한 패키지 설치
pip install -q gradio Pillow 2>/dev/null

echo ""
echo "============================================"
echo "  Demo Server Starting..."
echo "  URL: http://localhost:7860"
echo "============================================"
echo ""

python demo/app.py
