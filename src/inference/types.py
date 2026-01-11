"""추론 타입 정의 (외부 의존성 없음)"""

from dataclasses import dataclass
from typing import Any


@dataclass
class AnalysisResult:
    """분석 결과"""

    # VLM 분석 결과
    defect_type: str
    korean_name: str
    location: str
    severity: str
    reasoning: str

    # GraphRAG 추론 결과
    root_causes: list[dict[str, Any]]
    recommended_actions: list[dict[str, Any]]
    related_processes: list[dict[str, Any]]

    # 메타데이터
    confidence: float
    raw_response: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "defect_type": self.defect_type,
            "korean_name": self.korean_name,
            "location": self.location,
            "severity": self.severity,
            "reasoning": self.reasoning,
            "root_causes": self.root_causes,
            "recommended_actions": self.recommended_actions,
            "related_processes": self.related_processes,
            "confidence": self.confidence,
            "raw_response": self.raw_response,
        }

    def format_report(self) -> str:
        """포맷된 보고서 생성"""
        lines = [
            "=" * 60,
            "디스플레이 결함 분석 보고서",
            "=" * 60,
            "",
            f"**결함 유형**: {self.korean_name} ({self.defect_type})",
            f"**위치**: {self.location}",
            f"**심각도**: {self.severity}",
            f"**확신도**: {self.confidence * 100:.0f}%",
            "",
            "--- 추론 과정 ---",
            self.reasoning,
            "",
            "--- 추정 원인 ---",
        ]

        for i, cause in enumerate(self.root_causes[:3], 1):
            prob = cause.get("probability", 0) * 100
            lines.append(f"{i}. {cause.get('cause')} ({prob:.0f}%)")
            if cause.get("evidence"):
                lines.append(f"   근거: {cause.get('evidence')}")

        lines.extend(["", "--- 권장 조치 ---"])

        for i, action in enumerate(self.recommended_actions[:3], 1):
            lines.append(f"{i}. {action.get('action')}")
            lines.append(f"   - {action.get('description')}")
            lines.append(f"   - 우선순위: {action.get('priority')}")

        lines.extend(["", "=" * 60])

        return "\n".join(lines)
