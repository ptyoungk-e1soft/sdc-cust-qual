#!/bin/bash
# 로컬 환경에서 데모 실행 (Docker 없이)
# Neo4j 없이 인메모리 모드로 동작

echo "============================================"
echo "  디스플레이 결함 분석 시스템"
echo "  Local Demo (인메모리 모드)"
echo "============================================"

cd "$(dirname "$0")/.."

# 기존 프로세스 종료
echo ""
echo "기존 프로세스 종료 중..."
pkill -f "gradio\|app_full.py\|app.py" 2>/dev/null
sleep 2
echo "완료"

# 가상환경 활성화 (있으면)
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# 필요한 패키지 설치
echo ""
echo "패키지 설치 중..."
pip install -q gradio networkx matplotlib pillow pyyaml 2>/dev/null

echo ""
echo "============================================"
echo "  Demo Server Starting..."
echo "  URL: http://localhost:7860"
echo "============================================"
echo ""
echo "기능:"
echo "  1. 결함 분석 (시뮬레이션)"
echo "  2. GraphRAG 관리 (인메모리)"
echo "  3. 그래프 시각화"
echo ""

python demo/app_full.py
