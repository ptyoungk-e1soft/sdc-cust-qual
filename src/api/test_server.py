"""테스트용 API 서버 (VLM 모델 없이 동작)"""

import io
import os
import random
import time
from datetime import datetime
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from PIL import Image

from .schemas import (
    AnalysisResponse,
    BatchAnalysisResponse,
    StatisticsResponse,
    HealthResponse,
    KnowledgeBaseQuery,
    KnowledgeBaseResponse,
    RootCause,
    RecommendedAction,
    RelatedProcess,
)
from ..ontology.knowledge_base import KnowledgeBase
from ..ontology.reasoning import RootCauseReasoner, DefectEvidence
from ..data.constants import DEFECT_TYPES

# 지식 베이스 초기화
kb = KnowledgeBase(use_neo4j=False)
kb.load_defaults()
reasoner = RootCauseReasoner(kb)

# 분석 히스토리
_analysis_history: list[AnalysisResponse] = []

# FastAPI 앱
app = FastAPI(
    title="SDC Display Defect Analyzer (Test Mode)",
    description="Cosmos Reason VLM 기반 디스플레이 결함 분석 API - 테스트 모드",
    version="0.1.0-test",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@app.get("/version")
async def version():
    return {
        "name": "SDC Display Defect Analyzer",
        "version": "0.1.0-test",
        "mode": "test (no VLM model)",
    }


@app.get("/api/v1/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=False,  # 테스트 모드에서는 모델 없음
        knowledge_base_connected=True,
        version="0.1.0-test",
        timestamp=datetime.now(),
    )


@app.post("/api/v1/analyze", response_model=AnalysisResponse, tags=["Analysis"])
async def analyze_image(
    file: Annotated[UploadFile, File(description="분석할 이미지 파일")],
    additional_context: Annotated[str | None, Form()] = None,
    include_reasoning: Annotated[bool, Form()] = True,
):
    """이미지 결함 분석 (Mock 모드 - 랜덤 결함 유형 반환)"""
    start_time = time.time()

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        if image.mode != "RGB":
            image = image.convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"이미지 로드 실패: {str(e)}")

    # Mock: 랜덤 결함 유형 선택
    defect_type = random.choice(list(DEFECT_TYPES.keys()))
    locations = ["좌측 상단", "중앙 상단", "우측 상단", "좌측", "중앙부", "우측", "좌측 하단", "중앙 하단", "우측 하단"]
    severities = ["low", "medium", "high", "critical"]

    # GraphRAG로 근본원인 추론
    evidence = DefectEvidence(
        defect_type=defect_type,
        location=random.choice(locations),
        severity=random.choice(severities),
    )
    result = reasoner.reason(evidence)

    processing_time = (time.time() - start_time) * 1000

    response = AnalysisResponse(
        defect_type=result.defect_type,
        korean_name=result.korean_name,
        location=evidence.location,
        severity=evidence.severity,
        confidence=result.confidence,
        reasoning="\n".join(result.reasoning_chain) if include_reasoning else None,
        root_causes=[
            RootCause(
                cause=c.get("cause", ""),
                category=c.get("category"),
                description=c.get("description"),
                probability=c.get("probability", 0),
                evidence=c.get("evidence"),
            )
            for c in result.root_causes
        ],
        recommended_actions=[
            RecommendedAction(
                action=a.get("action", ""),
                description=a.get("description"),
                priority=a.get("priority", "medium"),
                effectiveness=a.get("effectiveness", 0),
                for_cause=a.get("for_cause"),
            )
            for a in result.recommended_actions
        ],
        related_processes=[
            RelatedProcess(
                process=p.get("process", ""),
                process_name=p.get("process_name"),
                frequency=p.get("frequency", "occasional"),
                sequence=p.get("sequence"),
            )
            for p in result.related_processes
        ],
        raw_response=f"[Mock] Detected: {result.korean_name}",
        processing_time_ms=processing_time,
        timestamp=datetime.now(),
    )

    _analysis_history.append(response)
    return response


@app.get("/api/v1/statistics", response_model=StatisticsResponse, tags=["Statistics"])
async def get_statistics():
    if not _analysis_history:
        return StatisticsResponse(
            total_analyzed=0,
            by_type={},
            by_severity={},
            common_causes={},
            avg_confidence=0,
        )

    type_counts = {}
    severity_counts = {}
    cause_counts = {}
    total_confidence = 0

    for result in _analysis_history:
        type_counts[result.korean_name] = type_counts.get(result.korean_name, 0) + 1
        severity_counts[result.severity] = severity_counts.get(result.severity, 0) + 1
        total_confidence += result.confidence

        for cause in result.root_causes:
            cause_counts[cause.cause] = cause_counts.get(cause.cause, 0) + 1

    return StatisticsResponse(
        total_analyzed=len(_analysis_history),
        by_type=type_counts,
        by_severity=severity_counts,
        common_causes=dict(sorted(cause_counts.items(), key=lambda x: x[1], reverse=True)[:5]),
        avg_confidence=total_confidence / len(_analysis_history),
    )


@app.post("/api/v1/knowledge/query", response_model=KnowledgeBaseResponse, tags=["Knowledge Base"])
async def query_knowledge_base(query: KnowledgeBaseQuery):
    analysis = kb.analyze_defect(query.defect_type)

    return KnowledgeBaseResponse(
        defect_type=query.defect_type,
        root_causes=[
            RootCause(
                cause=c.get("cause", ""),
                category=c.get("category"),
                description=c.get("description"),
                probability=c.get("probability", 0),
                evidence=c.get("evidence"),
            )
            for c in analysis.get("root_causes", [])[:query.limit]
        ],
        recommended_actions=[
            RecommendedAction(
                action=a.get("action", ""),
                description=a.get("description"),
                priority=a.get("priority", "medium"),
                effectiveness=a.get("effectiveness", 0),
                for_cause=a.get("for_cause"),
            )
            for a in analysis.get("recommended_actions", [])[:query.limit]
        ],
        related_processes=[
            RelatedProcess(
                process=p.get("process", ""),
                process_name=p.get("process_name"),
                frequency=p.get("frequency", "occasional"),
                sequence=p.get("sequence"),
            )
            for p in analysis.get("related_processes", [])
        ],
    )


@app.delete("/api/v1/history", tags=["History"])
async def clear_history():
    global _analysis_history
    count = len(_analysis_history)
    _analysis_history = []
    return {"message": f"{count}개 기록 삭제됨"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
