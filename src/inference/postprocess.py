"""후처리 모듈"""

import json
from dataclasses import dataclass
from typing import Any

from .types import AnalysisResult


@dataclass
class FormattedOutput:
    """포맷된 출력"""

    text: str
    json_data: dict[str, Any]
    html: str | None = None


class OutputFormatter:
    """출력 포맷터"""

    def __init__(self, language: str = "ko"):
        self.language = language

    def format_result(
        self,
        result: AnalysisResult,
        include_reasoning: bool = True,
        format_type: str = "structured",
    ) -> FormattedOutput:
        """결과 포맷팅"""
        json_data = result.to_dict()

        if format_type == "text":
            text = self._format_text(result, include_reasoning)
        elif format_type == "markdown":
            text = self._format_markdown(result, include_reasoning)
        else:
            text = self._format_structured(result, include_reasoning)

        html = self._format_html(result, include_reasoning)

        return FormattedOutput(
            text=text,
            json_data=json_data,
            html=html,
        )

    def _format_text(self, result: AnalysisResult, include_reasoning: bool) -> str:
        """텍스트 포맷"""
        lines = [
            f"결함 유형: {result.korean_name}",
            f"위치: {result.location}",
            f"심각도: {result.severity}",
            f"확신도: {result.confidence * 100:.0f}%",
        ]

        if include_reasoning and result.reasoning:
            lines.extend(["", "추론:", result.reasoning])

        if result.root_causes:
            lines.append("")
            lines.append("추정 원인:")
            for cause in result.root_causes[:3]:
                prob = cause.get("probability", 0) * 100
                lines.append(f"  - {cause.get('cause')} ({prob:.0f}%)")

        if result.recommended_actions:
            lines.append("")
            lines.append("권장 조치:")
            for action in result.recommended_actions[:3]:
                lines.append(f"  - {action.get('action')}: {action.get('description')}")

        return "\n".join(lines)

    def _format_structured(self, result: AnalysisResult, include_reasoning: bool) -> str:
        """구조화된 포맷"""
        lines = [
            "<think>" if include_reasoning else "",
            result.reasoning if include_reasoning else "",
            "</think>" if include_reasoning else "",
            "<answer>",
            f"결함 유형: {result.korean_name}",
            f"위치: {result.location}",
            f"심각도: {result.severity}",
        ]

        if result.root_causes:
            cause = result.root_causes[0]
            lines.append(f"가능한 원인: {cause.get('cause')}")

        if result.recommended_actions:
            action = result.recommended_actions[0]
            lines.append(f"권장 조치: {action.get('action')}")

        lines.append("</answer>")

        return "\n".join(filter(None, lines))

    def _format_markdown(self, result: AnalysisResult, include_reasoning: bool) -> str:
        """마크다운 포맷"""
        lines = [
            "## 결함 분석 결과",
            "",
            f"**결함 유형**: {result.korean_name} (`{result.defect_type}`)",
            f"**위치**: {result.location}",
            f"**심각도**: {result.severity}",
            f"**확신도**: {result.confidence * 100:.0f}%",
        ]

        if include_reasoning and result.reasoning:
            lines.extend([
                "",
                "### 분석 과정",
                result.reasoning,
            ])

        if result.root_causes:
            lines.extend(["", "### 추정 원인"])
            for i, cause in enumerate(result.root_causes[:3], 1):
                prob = cause.get("probability", 0) * 100
                lines.append(f"{i}. **{cause.get('cause')}** ({prob:.0f}%)")
                if cause.get("evidence"):
                    lines.append(f"   - 근거: {cause.get('evidence')}")

        if result.recommended_actions:
            lines.extend(["", "### 권장 조치"])
            for i, action in enumerate(result.recommended_actions[:3], 1):
                lines.append(f"{i}. **{action.get('action')}**")
                lines.append(f"   - {action.get('description')}")
                lines.append(f"   - 우선순위: `{action.get('priority')}`")

        return "\n".join(lines)

    def _format_html(self, result: AnalysisResult, include_reasoning: bool) -> str:
        """HTML 포맷"""
        severity_colors = {
            "low": "#4CAF50",
            "medium": "#FFC107",
            "high": "#FF9800",
            "critical": "#F44336",
        }

        severity_color = severity_colors.get(result.severity.lower(), "#9E9E9E")

        html = f"""
<div class="defect-analysis">
    <h2>결함 분석 결과</h2>

    <div class="summary">
        <div class="field">
            <label>결함 유형</label>
            <span class="value">{result.korean_name}</span>
        </div>
        <div class="field">
            <label>위치</label>
            <span class="value">{result.location}</span>
        </div>
        <div class="field">
            <label>심각도</label>
            <span class="severity" style="background-color: {severity_color}">
                {result.severity}
            </span>
        </div>
        <div class="field">
            <label>확신도</label>
            <span class="value">{result.confidence * 100:.0f}%</span>
        </div>
    </div>
"""

        if include_reasoning and result.reasoning:
            html += f"""
    <div class="reasoning">
        <h3>분석 과정</h3>
        <p>{result.reasoning}</p>
    </div>
"""

        if result.root_causes:
            html += """
    <div class="causes">
        <h3>추정 원인</h3>
        <ul>
"""
            for cause in result.root_causes[:3]:
                prob = cause.get("probability", 0) * 100
                html += f"""
            <li>
                <strong>{cause.get('cause')}</strong> ({prob:.0f}%)
                <br><small>{cause.get('evidence', '')}</small>
            </li>
"""
            html += """
        </ul>
    </div>
"""

        if result.recommended_actions:
            html += """
    <div class="actions">
        <h3>권장 조치</h3>
        <ul>
"""
            for action in result.recommended_actions[:3]:
                html += f"""
            <li>
                <strong>{action.get('action')}</strong>
                <br><span>{action.get('description')}</span>
                <br><small>우선순위: {action.get('priority')}</small>
            </li>
"""
            html += """
        </ul>
    </div>
"""

        html += """
</div>
"""
        return html


class BatchReportGenerator:
    """배치 분석 보고서 생성기"""

    def __init__(self, formatter: OutputFormatter | None = None):
        self.formatter = formatter or OutputFormatter()

    def generate_summary_report(
        self,
        results: list[AnalysisResult],
        title: str = "디스플레이 결함 분석 보고서",
    ) -> str:
        """요약 보고서 생성"""
        lines = [
            f"# {title}",
            "",
            "## 요약",
            f"- 총 분석 건수: {len(results)}건",
        ]

        # 유형별 통계
        type_counts: dict[str, int] = {}
        severity_counts: dict[str, int] = {}
        total_confidence = 0

        for result in results:
            type_counts[result.korean_name] = type_counts.get(result.korean_name, 0) + 1
            severity_counts[result.severity] = severity_counts.get(result.severity, 0) + 1
            total_confidence += result.confidence

        avg_confidence = total_confidence / len(results) if results else 0

        lines.extend([
            f"- 평균 확신도: {avg_confidence * 100:.0f}%",
            "",
            "### 결함 유형별 분포",
        ])

        for dtype, count in sorted(type_counts.items(), key=lambda x: x[1], reverse=True):
            pct = count / len(results) * 100
            lines.append(f"- {dtype}: {count}건 ({pct:.1f}%)")

        lines.extend(["", "### 심각도별 분포"])

        for severity, count in sorted(severity_counts.items()):
            pct = count / len(results) * 100
            lines.append(f"- {severity}: {count}건 ({pct:.1f}%)")

        # 상위 원인
        cause_counts: dict[str, int] = {}
        for result in results:
            for cause in result.root_causes:
                cause_name = cause.get("cause", "")
                cause_counts[cause_name] = cause_counts.get(cause_name, 0) + 1

        if cause_counts:
            lines.extend(["", "### 주요 추정 원인"])
            for cause, count in sorted(cause_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
                lines.append(f"- {cause}: {count}건")

        lines.extend(["", "---", "", "## 상세 분석 결과", ""])

        # 개별 결과
        for i, result in enumerate(results, 1):
            lines.append(f"### {i}. {result.korean_name}")
            lines.append(f"- 위치: {result.location}")
            lines.append(f"- 심각도: {result.severity}")
            if result.root_causes:
                lines.append(f"- 주요 원인: {result.root_causes[0].get('cause')}")
            lines.append("")

        return "\n".join(lines)

    def export_to_json(self, results: list[AnalysisResult]) -> str:
        """JSON 형식으로 내보내기"""
        data = {
            "total": len(results),
            "results": [r.to_dict() for r in results],
        }
        return json.dumps(data, ensure_ascii=False, indent=2)

    def export_to_csv(self, results: list[AnalysisResult]) -> str:
        """CSV 형식으로 내보내기"""
        import csv
        import io

        output = io.StringIO()
        writer = csv.writer(output)

        # 헤더
        writer.writerow([
            "defect_type",
            "korean_name",
            "location",
            "severity",
            "confidence",
            "top_cause",
            "top_action",
        ])

        # 데이터
        for result in results:
            top_cause = result.root_causes[0].get("cause") if result.root_causes else ""
            top_action = result.recommended_actions[0].get("action") if result.recommended_actions else ""

            writer.writerow([
                result.defect_type,
                result.korean_name,
                result.location,
                result.severity,
                f"{result.confidence:.2f}",
                top_cause,
                top_action,
            ])

        return output.getvalue()
