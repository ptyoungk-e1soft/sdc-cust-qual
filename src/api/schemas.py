"""Pydantic 스키마"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class DefectType(str, Enum):
    """결함 유형"""

    DEAD_PIXEL = "dead_pixel"
    BRIGHT_SPOT = "bright_spot"
    LINE_DEFECT = "line_defect"
    MURA = "mura"
    SCRATCH = "scratch"
    PARTICLE = "particle"
    CUSTOM = "custom"


class SeverityLevel(str, Enum):
    """심각도 수준"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RootCause(BaseModel):
    """근본 원인"""

    cause: str = Field(..., description="원인명")
    category: str | None = Field(None, description="원인 분류")
    description: str | None = Field(None, description="설명")
    probability: float = Field(..., ge=0, le=1, description="확률")
    evidence: str | None = Field(None, description="근거")


class RecommendedAction(BaseModel):
    """권장 조치"""

    action: str = Field(..., description="조치명")
    description: str | None = Field(None, description="설명")
    priority: str = Field("medium", description="우선순위")
    effectiveness: float = Field(..., ge=0, le=1, description="효과성")
    for_cause: str | None = Field(None, description="대상 원인")


class RelatedProcess(BaseModel):
    """관련 공정"""

    process: str = Field(..., description="공정명")
    process_name: str | None = Field(None, description="영문명")
    frequency: str = Field("occasional", description="발생 빈도")
    sequence: int | None = Field(None, description="공정 순서")


class AnalysisRequest(BaseModel):
    """분석 요청"""

    additional_context: str | None = Field(None, description="추가 컨텍스트")
    include_reasoning: bool = Field(True, description="추론 과정 포함 여부")
    format: str = Field("structured", description="출력 형식 (structured, text, json)")


class AnalysisResponse(BaseModel):
    """분석 응답"""

    # 기본 정보
    defect_type: str = Field(..., description="결함 유형 코드")
    korean_name: str = Field(..., description="결함 유형 한글명")
    location: str = Field(..., description="결함 위치")
    severity: str = Field(..., description="심각도")
    confidence: float = Field(..., ge=0, le=1, description="확신도")

    # 분석 결과
    reasoning: str | None = Field(None, description="추론 과정")
    root_causes: list[RootCause] = Field(default_factory=list, description="추정 원인")
    recommended_actions: list[RecommendedAction] = Field(
        default_factory=list, description="권장 조치"
    )
    related_processes: list[RelatedProcess] = Field(
        default_factory=list, description="관련 공정"
    )

    # 메타데이터
    raw_response: str | None = Field(None, description="원본 VLM 응답")
    processing_time_ms: float | None = Field(None, description="처리 시간 (ms)")
    timestamp: datetime = Field(default_factory=datetime.now, description="분석 시간")


class BatchAnalysisRequest(BaseModel):
    """배치 분석 요청"""

    additional_contexts: list[str] | None = Field(None, description="추가 컨텍스트 목록")
    include_reasoning: bool = Field(True, description="추론 과정 포함 여부")


class BatchAnalysisResponse(BaseModel):
    """배치 분석 응답"""

    total: int = Field(..., description="총 분석 건수")
    results: list[AnalysisResponse] = Field(..., description="분석 결과 목록")
    processing_time_ms: float | None = Field(None, description="총 처리 시간 (ms)")


class StatisticsResponse(BaseModel):
    """통계 응답"""

    total_analyzed: int = Field(..., description="총 분석 건수")
    by_type: dict[str, int] = Field(default_factory=dict, description="유형별 분포")
    by_severity: dict[str, int] = Field(default_factory=dict, description="심각도별 분포")
    common_causes: dict[str, int] = Field(default_factory=dict, description="주요 원인")
    avg_confidence: float = Field(..., description="평균 확신도")


class HealthResponse(BaseModel):
    """헬스체크 응답"""

    status: str = Field(..., description="서비스 상태")
    model_loaded: bool = Field(..., description="모델 로드 여부")
    knowledge_base_connected: bool = Field(..., description="지식베이스 연결 여부")
    version: str = Field(..., description="API 버전")
    timestamp: datetime = Field(default_factory=datetime.now)


class ErrorResponse(BaseModel):
    """에러 응답"""

    error: str = Field(..., description="에러 메시지")
    detail: str | None = Field(None, description="상세 정보")
    timestamp: datetime = Field(default_factory=datetime.now)


class KnowledgeBaseQuery(BaseModel):
    """지식베이스 쿼리"""

    defect_type: str = Field(..., description="결함 유형")
    limit: int = Field(5, ge=1, le=20, description="결과 개수 제한")


class KnowledgeBaseResponse(BaseModel):
    """지식베이스 응답"""

    defect_type: str = Field(..., description="결함 유형")
    root_causes: list[RootCause] = Field(default_factory=list)
    recommended_actions: list[RecommendedAction] = Field(default_factory=list)
    related_processes: list[RelatedProcess] = Field(default_factory=list)
