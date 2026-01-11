#!/bin/bash
# 디스플레이 결함 분석 시스템 - 전체 데모 실행
# Neo4j + GraphRAG + 시각화 포함

echo "============================================"
echo "  디스플레이 결함 분석 시스템"
echo "  Full Demo (Neo4j + GraphRAG)"
echo "============================================"

cd "$(dirname "$0")"

# 기존 프로세스 종료
echo ""
echo "기존 프로세스 종료 중..."
pkill -f "gradio\|app_full.py\|app.py" 2>/dev/null
sleep 2
echo "완료"

# Docker Compose로 실행
if command -v docker-compose &> /dev/null; then
    COMPOSE_CMD="docker-compose"
elif command -v docker &> /dev/null && docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
else
    echo "Docker Compose가 설치되어 있지 않습니다."
    exit 1
fi

echo ""
echo "서비스 시작 중..."
echo "  - Neo4j: http://localhost:7474 (neo4j/password)"
echo "  - Demo UI: http://localhost:7860"
echo ""

$COMPOSE_CMD -f docker-compose.demo.yml up --build

# 종료 시 정리
trap "$COMPOSE_CMD -f docker-compose.demo.yml down" EXIT
