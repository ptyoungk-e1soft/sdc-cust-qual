"""근본원인 추론 엔진"""

from dataclasses import dataclass
from typing import Any

from .knowledge_base import KnowledgeBase
from .schema import DefectType, SeverityLevel


@dataclass
class DefectEvidence:
    """결함 증거 데이터"""

    defect_type: str
    location: str
    severity: str
    visual_features: list[str] | None = None
    additional_context: str | None = None


@dataclass
class ReasoningResult:
    """추론 결과"""

    defect_type: str
    korean_name: str
    confidence: float
    root_causes: list[dict[str, Any]]
    recommended_actions: list[dict[str, Any]]
    related_processes: list[dict[str, Any]]
    reasoning_chain: list[str]


class RootCauseReasoner:
    """근본원인 추론 엔진"""

    def __init__(self, knowledge_base: KnowledgeBase):
        self.kb = knowledge_base

    def reason(self, evidence: DefectEvidence) -> ReasoningResult:
        """증거를 기반으로 근본원인 추론"""
        defect_type = evidence.defect_type

        # 지식 베이스에서 분석 정보 조회
        analysis = self.kb.analyze_defect(defect_type)

        # 추론 체인 구성
        reasoning_chain = self._build_reasoning_chain(evidence, analysis)

        # 확신도 계산
        confidence = self._calculate_confidence(evidence, analysis)

        # 결과 구성
        try:
            korean_name = DefectType(defect_type).korean_name
        except ValueError:
            korean_name = defect_type

        return ReasoningResult(
            defect_type=defect_type,
            korean_name=korean_name,
            confidence=confidence,
            root_causes=analysis.get("root_causes", []),
            recommended_actions=analysis.get("recommended_actions", []),
            related_processes=analysis.get("related_processes", []),
            reasoning_chain=reasoning_chain,
        )

    def _build_reasoning_chain(
        self,
        evidence: DefectEvidence,
        analysis: dict[str, Any],
    ) -> list[str]:
        """추론 체인 구성"""
        chain = []

        # 1. 결함 감지
        chain.append(f"1. {evidence.location} 영역에서 {evidence.defect_type} 유형의 결함이 감지되었습니다.")

        # 2. 심각도 평가
        severity_desc = self._get_severity_description(evidence.severity)
        chain.append(f"2. 결함 심각도는 {severity_desc}로 평가됩니다.")

        # 3. 원인 분석
        root_causes = analysis.get("root_causes", [])
        if root_causes:
            top_cause = root_causes[0]
            prob = top_cause.get("probability", 0) * 100
            chain.append(
                f"3. 가장 유력한 원인은 '{top_cause.get('cause')}'이며, "
                f"확률은 {prob:.0f}%입니다."
            )
            if top_cause.get("evidence"):
                chain.append(f"   근거: {top_cause.get('evidence')}")

        # 4. 추가 원인 검토
        if len(root_causes) > 1:
            other_causes = [c.get("cause") for c in root_causes[1:3]]
            chain.append(f"4. 추가 검토가 필요한 원인: {', '.join(other_causes)}")

        # 5. 조치 권고
        actions = analysis.get("recommended_actions", [])
        if actions:
            top_action = actions[0]
            chain.append(
                f"5. 우선 권장 조치: '{top_action.get('action')}' - {top_action.get('description')}"
            )

        return chain

    def _get_severity_description(self, severity: str) -> str:
        """심각도 설명"""
        descriptions = {
            "low": "경미 (품질에 영향 없음)",
            "medium": "보통 (검토 필요)",
            "high": "심각 (즉시 조치 필요)",
            "critical": "치명적 (라인 중단 검토)",
        }
        return descriptions.get(severity.lower(), severity)

    def _calculate_confidence(
        self,
        evidence: DefectEvidence,
        analysis: dict[str, Any],
    ) -> float:
        """추론 확신도 계산"""
        confidence = 0.5  # 기본값

        # 근본 원인이 있으면 확신도 상승
        root_causes = analysis.get("root_causes", [])
        if root_causes:
            top_prob = root_causes[0].get("probability", 0)
            confidence = max(confidence, top_prob)

        # 관련 공정 정보가 있으면 확신도 상승
        if analysis.get("related_processes"):
            confidence = min(1.0, confidence + 0.1)

        # 권장 조치가 있으면 확신도 상승
        if analysis.get("recommended_actions"):
            confidence = min(1.0, confidence + 0.05)

        return confidence

    def format_reasoning_output(self, result: ReasoningResult) -> str:
        """추론 결과를 포맷팅된 문자열로 변환"""
        lines = [
            f"<think>",
            "이미지 분석 결과를 바탕으로 결함을 추론합니다.",
            "",
        ]

        for step in result.reasoning_chain:
            lines.append(step)

        lines.extend([
            "",
            "</think>",
            "<answer>",
            f"결함 유형: {result.korean_name}",
            f"확신도: {result.confidence * 100:.0f}%",
            "",
        ])

        if result.root_causes:
            top_cause = result.root_causes[0]
            lines.append(f"가능한 원인: {top_cause.get('cause')}")
            if top_cause.get("evidence"):
                lines.append(f"  - 근거: {top_cause.get('evidence')}")

        if result.recommended_actions:
            top_action = result.recommended_actions[0]
            lines.append(f"권장 조치: {top_action.get('action')}")
            lines.append(f"  - {top_action.get('description')}")

        lines.append("</answer>")

        return "\n".join(lines)


class MultiDefectAnalyzer:
    """다중 결함 분석기"""

    def __init__(self, knowledge_base: KnowledgeBase):
        self.kb = knowledge_base
        self.reasoner = RootCauseReasoner(knowledge_base)

    def analyze_multiple(self, defects: list[DefectEvidence]) -> dict[str, Any]:
        """복수 결함 분석"""
        results = []
        for defect in defects:
            result = self.reasoner.reason(defect)
            results.append(result)

        # 공통 원인 분석
        common_causes = self._find_common_causes(results)

        # 우선순위 정렬
        prioritized = self._prioritize_actions(results)

        return {
            "individual_results": results,
            "common_causes": common_causes,
            "prioritized_actions": prioritized,
            "summary": self._generate_summary(results, common_causes),
        }

    def _find_common_causes(self, results: list[ReasoningResult]) -> list[dict[str, Any]]:
        """공통 원인 찾기"""
        cause_counts: dict[str, dict[str, Any]] = {}

        for result in results:
            for cause in result.root_causes:
                cause_name = cause.get("cause", "")
                if cause_name not in cause_counts:
                    cause_counts[cause_name] = {
                        "cause": cause_name,
                        "category": cause.get("category"),
                        "count": 0,
                        "total_probability": 0,
                    }
                cause_counts[cause_name]["count"] += 1
                cause_counts[cause_name]["total_probability"] += cause.get("probability", 0)

        # 2개 이상의 결함에서 나타나는 원인
        common = [c for c in cause_counts.values() if c["count"] >= 2]
        common.sort(key=lambda x: x["total_probability"], reverse=True)

        return common

    def _prioritize_actions(self, results: list[ReasoningResult]) -> list[dict[str, Any]]:
        """조치 우선순위 정렬"""
        action_scores: dict[str, dict[str, Any]] = {}

        priority_weights = {
            "immediate": 4,
            "high": 3,
            "medium": 2,
            "low": 1,
        }

        for result in results:
            for action in result.recommended_actions:
                action_name = action.get("action", "")
                if action_name not in action_scores:
                    action_scores[action_name] = {
                        "action": action_name,
                        "description": action.get("description"),
                        "score": 0,
                        "for_defects": [],
                    }

                priority = action.get("priority", "medium")
                effectiveness = action.get("effectiveness", 0.5)
                score = priority_weights.get(priority, 2) * effectiveness

                action_scores[action_name]["score"] += score
                action_scores[action_name]["for_defects"].append(result.defect_type)

        prioritized = list(action_scores.values())
        prioritized.sort(key=lambda x: x["score"], reverse=True)

        return prioritized

    def _generate_summary(
        self,
        results: list[ReasoningResult],
        common_causes: list[dict[str, Any]],
    ) -> str:
        """요약 생성"""
        defect_types = [r.korean_name for r in results]
        summary_lines = [
            f"총 {len(results)}개의 결함이 분석되었습니다: {', '.join(defect_types)}",
        ]

        if common_causes:
            common_names = [c["cause"] for c in common_causes[:3]]
            summary_lines.append(
                f"공통 추정 원인: {', '.join(common_names)}"
            )

        # 심각도별 분류
        severity_counts = {}
        for result in results:
            # 추정 (결함 유형에서 심각도 유추 가능)
            severity_counts["medium"] = severity_counts.get("medium", 0) + 1

        return " ".join(summary_lines)
