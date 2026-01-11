"""API 라우트"""

import io
import time
from datetime import datetime
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, Depends
from PIL import Image

from .schemas import (
    AnalysisRequest,
    AnalysisResponse,
    BatchAnalysisResponse,
    StatisticsResponse,
    HealthResponse,
    ErrorResponse,
    KnowledgeBaseQuery,
    KnowledgeBaseResponse,
    RootCause,
    RecommendedAction,
    RelatedProcess,
)
from ..inference.pipeline import InferencePipeline, PipelineConfig

# 라우터 생성
router = APIRouter()

# 전역 파이프라인 (서버 시작 시 초기화)
_pipeline: InferencePipeline | None = None
_analysis_history: list[AnalysisResponse] = []


def get_pipeline() -> InferencePipeline:
    """파이프라인 의존성"""
    global _pipeline
    if _pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="서비스가 아직 초기화되지 않았습니다.",
        )
    return _pipeline


def init_pipeline(config: PipelineConfig | None = None) -> None:
    """파이프라인 초기화"""
    global _pipeline
    _pipeline = InferencePipeline(config)
    _pipeline.initialize()


def shutdown_pipeline() -> None:
    """파이프라인 종료"""
    global _pipeline
    if _pipeline:
        _pipeline.shutdown()
        _pipeline = None


@router.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """서비스 상태 확인"""
    global _pipeline

    return HealthResponse(
        status="healthy" if _pipeline else "initializing",
        model_loaded=_pipeline is not None and _pipeline.analyzer is not None,
        knowledge_base_connected=_pipeline is not None
        and _pipeline.analyzer is not None
        and _pipeline.analyzer.kb is not None,
        version="0.1.0",
        timestamp=datetime.now(),
    )


@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    tags=["Analysis"],
)
async def analyze_image(
    file: Annotated[UploadFile, File(description="분석할 이미지 파일")],
    additional_context: Annotated[str | None, Form()] = None,
    include_reasoning: Annotated[bool, Form()] = True,
    pipeline: InferencePipeline = Depends(get_pipeline),
):
    """단일 이미지 결함 분석"""
    start_time = time.time()

    try:
        # 이미지 로드
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        if image.mode != "RGB":
            image = image.convert("RGB")

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"이미지 로드 실패: {str(e)}",
        )

    try:
        # 분석 수행
        result = pipeline.process(
            image=image,
            additional_context=additional_context,
        )

        processing_time = (time.time() - start_time) * 1000

        # 응답 생성
        response = AnalysisResponse(
            defect_type=result.defect_type,
            korean_name=result.korean_name,
            location=result.location,
            severity=result.severity,
            confidence=result.confidence,
            reasoning=result.reasoning if include_reasoning else None,
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
            raw_response=result.raw_response,
            processing_time_ms=processing_time,
            timestamp=datetime.now(),
        )

        # 히스토리에 추가
        _analysis_history.append(response)

        return response

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"분석 실패: {str(e)}",
        )


@router.post(
    "/analyze/batch",
    response_model=BatchAnalysisResponse,
    tags=["Analysis"],
)
async def analyze_batch(
    files: Annotated[list[UploadFile], File(description="분석할 이미지 파일들")],
    additional_contexts: Annotated[str | None, Form()] = None,
    include_reasoning: Annotated[bool, Form()] = True,
    pipeline: InferencePipeline = Depends(get_pipeline),
):
    """배치 이미지 결함 분석"""
    start_time = time.time()

    # 컨텍스트 파싱 (JSON 문자열로 전달된 경우)
    contexts = None
    if additional_contexts:
        try:
            import json

            contexts = json.loads(additional_contexts)
        except json.JSONDecodeError:
            contexts = [additional_contexts] * len(files)

    images = []
    for file in files:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        if image.mode != "RGB":
            image = image.convert("RGB")
        images.append(image)

    results = pipeline.process_batch(images, contexts)

    processing_time = (time.time() - start_time) * 1000

    responses = []
    for result in results:
        response = AnalysisResponse(
            defect_type=result.defect_type,
            korean_name=result.korean_name,
            location=result.location,
            severity=result.severity,
            confidence=result.confidence,
            reasoning=result.reasoning if include_reasoning else None,
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
            processing_time_ms=None,
            timestamp=datetime.now(),
        )
        responses.append(response)
        _analysis_history.append(response)

    return BatchAnalysisResponse(
        total=len(responses),
        results=responses,
        processing_time_ms=processing_time,
    )


@router.get("/statistics", response_model=StatisticsResponse, tags=["Statistics"])
async def get_statistics():
    """분석 통계 조회"""
    if not _analysis_history:
        return StatisticsResponse(
            total_analyzed=0,
            by_type={},
            by_severity={},
            common_causes={},
            avg_confidence=0,
        )

    type_counts: dict[str, int] = {}
    severity_counts: dict[str, int] = {}
    cause_counts: dict[str, int] = {}
    total_confidence = 0

    for result in _analysis_history:
        type_counts[result.korean_name] = type_counts.get(result.korean_name, 0) + 1
        severity_counts[result.severity] = severity_counts.get(result.severity, 0) + 1
        total_confidence += result.confidence

        for cause in result.root_causes:
            cause_name = cause.cause
            cause_counts[cause_name] = cause_counts.get(cause_name, 0) + 1

    return StatisticsResponse(
        total_analyzed=len(_analysis_history),
        by_type=type_counts,
        by_severity=severity_counts,
        common_causes=dict(sorted(cause_counts.items(), key=lambda x: x[1], reverse=True)[:5]),
        avg_confidence=total_confidence / len(_analysis_history),
    )


@router.post("/knowledge/query", response_model=KnowledgeBaseResponse, tags=["Knowledge Base"])
async def query_knowledge_base(
    query: KnowledgeBaseQuery,
    pipeline: InferencePipeline = Depends(get_pipeline),
):
    """지식베이스 쿼리"""
    if not pipeline.analyzer or not pipeline.analyzer.kb:
        raise HTTPException(
            status_code=503,
            detail="지식베이스가 초기화되지 않았습니다.",
        )

    analysis = pipeline.analyzer.kb.analyze_defect(query.defect_type)

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


@router.delete("/cache", tags=["Cache"])
async def clear_cache(pipeline: InferencePipeline = Depends(get_pipeline)):
    """캐시 초기화"""
    pipeline.clear_cache()
    return {"message": "캐시가 초기화되었습니다."}


@router.get("/cache/stats", tags=["Cache"])
async def get_cache_stats(pipeline: InferencePipeline = Depends(get_pipeline)):
    """캐시 통계"""
    return pipeline.get_cache_stats()
