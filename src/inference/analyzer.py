"""결함 분석기"""

from pathlib import Path
from typing import Any

from PIL import Image

from .types import AnalysisResult
from ..model.cosmos_wrapper import CosmosReasonWrapper
from ..ontology.knowledge_base import KnowledgeBase
from ..ontology.reasoning import RootCauseReasoner, DefectEvidence


class DefectAnalyzer:
    """디스플레이 결함 분석기

    VLM과 GraphRAG를 결합하여 결함 분석 수행
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        ontology_config_path: str | Path | None = None,
        use_neo4j: bool = False,
        quantize: bool = False,
    ):
        self.model_path = model_path
        self.ontology_config_path = ontology_config_path
        self.use_neo4j = use_neo4j
        self.quantize = quantize

        self.vlm: CosmosReasonWrapper | None = None
        self.kb: KnowledgeBase | None = None
        self.reasoner: RootCauseReasoner | None = None

        self._initialized = False

    def initialize(self) -> None:
        """분석기 초기화"""
        if self._initialized:
            return

        # VLM 모델 로드
        self.vlm = CosmosReasonWrapper(
            model_path=self.model_path,
            load_in_4bit=self.quantize,
        )
        self.vlm.load()

        # 지식 베이스 초기화
        self.kb = KnowledgeBase(
            config_path=self.ontology_config_path,
            use_neo4j=self.use_neo4j,
        )

        if self.use_neo4j:
            self.kb.connect()

        # 온톨로지 로드
        if self.ontology_config_path and Path(self.ontology_config_path).exists():
            self.kb.load_from_config(self.ontology_config_path)
        else:
            self.kb.load_defaults()

        # 추론 엔진 초기화
        self.reasoner = RootCauseReasoner(self.kb)

        self._initialized = True

    def shutdown(self) -> None:
        """분석기 종료"""
        if self.vlm:
            self.vlm.unload()
        if self.kb:
            self.kb.close()
        self._initialized = False

    def analyze(
        self,
        image: Image.Image | str | Path,
        additional_context: str | None = None,
    ) -> AnalysisResult:
        """이미지 분석 수행"""
        if not self._initialized:
            self.initialize()

        # VLM으로 결함 분석
        vlm_result = self.vlm.analyze_defect(
            image=image,
            additional_context=additional_context,
        )

        # 결함 유형 정규화
        defect_type = self._normalize_defect_type(vlm_result.get("defect_type", ""))

        # GraphRAG로 근본원인 추론
        evidence = DefectEvidence(
            defect_type=defect_type,
            location=vlm_result.get("location", ""),
            severity=vlm_result.get("severity", "medium"),
        )

        reasoning_result = self.reasoner.reason(evidence)

        # 결과 구성
        return AnalysisResult(
            defect_type=defect_type,
            korean_name=reasoning_result.korean_name,
            location=vlm_result.get("location", "확인 필요"),
            severity=vlm_result.get("severity", "medium"),
            reasoning=vlm_result.get("reasoning", ""),
            root_causes=reasoning_result.root_causes,
            recommended_actions=reasoning_result.recommended_actions,
            related_processes=reasoning_result.related_processes,
            confidence=reasoning_result.confidence,
            raw_response=vlm_result.get("raw_response", ""),
        )

    def analyze_batch(
        self,
        images: list[Image.Image | str | Path],
        additional_contexts: list[str] | None = None,
    ) -> list[AnalysisResult]:
        """배치 분석"""
        if additional_contexts is None:
            additional_contexts = [None] * len(images)

        results = []
        for image, context in zip(images, additional_contexts):
            result = self.analyze(image, context)
            results.append(result)

        return results

    def _normalize_defect_type(self, defect_type: str) -> str:
        """결함 유형 정규화"""
        # 한글/영문 매핑
        type_mapping = {
            "데드 픽셀": "dead_pixel",
            "dead pixel": "dead_pixel",
            "휘점": "bright_spot",
            "휘점 결함": "bright_spot",
            "bright spot": "bright_spot",
            "라인 결함": "line_defect",
            "라인": "line_defect",
            "line defect": "line_defect",
            "무라": "mura",
            "얼룩": "mura",
            "스크래치": "scratch",
            "scratch": "scratch",
            "긁힘": "scratch",
            "이물질": "particle",
            "particle": "particle",
            "파티클": "particle",
        }

        normalized = defect_type.lower().strip()

        for key, value in type_mapping.items():
            if key in normalized:
                return value

        # 매핑 실패 시 원본 반환 (또는 custom)
        if normalized:
            return normalized
        return "custom"

    def get_defect_statistics(self, results: list[AnalysisResult]) -> dict[str, Any]:
        """결함 통계 생성"""
        stats = {
            "total": len(results),
            "by_type": {},
            "by_severity": {},
            "common_causes": {},
            "avg_confidence": 0,
        }

        confidence_sum = 0

        for result in results:
            # 유형별 카운트
            dtype = result.defect_type
            stats["by_type"][dtype] = stats["by_type"].get(dtype, 0) + 1

            # 심각도별 카운트
            severity = result.severity
            stats["by_severity"][severity] = stats["by_severity"].get(severity, 0) + 1

            # 원인 카운트
            for cause in result.root_causes:
                cause_name = cause.get("cause", "")
                stats["common_causes"][cause_name] = stats["common_causes"].get(cause_name, 0) + 1

            confidence_sum += result.confidence

        if results:
            stats["avg_confidence"] = confidence_sum / len(results)

        # 상위 원인 정렬
        stats["common_causes"] = dict(
            sorted(stats["common_causes"].items(), key=lambda x: x[1], reverse=True)[:5]
        )

        return stats
