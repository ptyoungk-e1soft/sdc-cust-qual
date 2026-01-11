"""추론 모듈 테스트"""

import pytest

from src.inference.postprocess import OutputFormatter, BatchReportGenerator
from src.inference.types import AnalysisResult


class TestOutputFormatter:
    """출력 포맷터 테스트"""

    def test_format_text(self):
        formatter = OutputFormatter()

        result = AnalysisResult(
            defect_type="dead_pixel",
            korean_name="데드 픽셀",
            location="중앙",
            severity="medium",
            reasoning="테스트 추론",
            root_causes=[{"cause": "TFT 결함", "probability": 0.8}],
            recommended_actions=[{"action": "장비 점검", "description": "설명"}],
            related_processes=[],
            confidence=0.85,
            raw_response="테스트",
        )

        output = formatter.format_result(result, format_type="text")

        assert "데드 픽셀" in output.text
        assert "중앙" in output.text
        assert output.json_data is not None

    def test_format_markdown(self):
        formatter = OutputFormatter()

        result = AnalysisResult(
            defect_type="mura",
            korean_name="무라",
            location="좌측",
            severity="high",
            reasoning="무라 감지됨",
            root_causes=[{"cause": "증착 결함", "probability": 0.7, "evidence": "근거"}],
            recommended_actions=[{"action": "공정 조정", "description": "파라미터 조정", "priority": "high"}],
            related_processes=[],
            confidence=0.75,
            raw_response="테스트",
        )

        output = formatter.format_result(result, format_type="markdown")

        assert "##" in output.text
        assert "무라" in output.text
        assert "**" in output.text

    def test_format_html(self):
        formatter = OutputFormatter()

        result = AnalysisResult(
            defect_type="scratch",
            korean_name="스크래치",
            location="우측 하단",
            severity="low",
            reasoning="긁힘 발견",
            root_causes=[],
            recommended_actions=[],
            related_processes=[],
            confidence=0.6,
            raw_response="테스트",
        )

        output = formatter.format_result(result)

        assert output.html is not None
        assert "<div" in output.html
        assert "스크래치" in output.html


class TestBatchReportGenerator:
    """배치 보고서 생성기 테스트"""

    def test_generate_summary(self):
        generator = BatchReportGenerator()

        results = [
            AnalysisResult(
                defect_type="dead_pixel",
                korean_name="데드 픽셀",
                location="중앙",
                severity="medium",
                reasoning="",
                root_causes=[{"cause": "TFT 결함", "probability": 0.8}],
                recommended_actions=[],
                related_processes=[],
                confidence=0.85,
                raw_response="",
            ),
            AnalysisResult(
                defect_type="mura",
                korean_name="무라",
                location="좌측",
                severity="high",
                reasoning="",
                root_causes=[{"cause": "증착 결함", "probability": 0.7}],
                recommended_actions=[],
                related_processes=[],
                confidence=0.75,
                raw_response="",
            ),
        ]

        report = generator.generate_summary_report(results)

        assert "총 분석 건수: 2건" in report
        assert "데드 픽셀" in report
        assert "무라" in report

    def test_export_to_json(self):
        generator = BatchReportGenerator()

        results = [
            AnalysisResult(
                defect_type="particle",
                korean_name="이물질",
                location="중앙",
                severity="high",
                reasoning="",
                root_causes=[],
                recommended_actions=[],
                related_processes=[],
                confidence=0.9,
                raw_response="",
            ),
        ]

        json_output = generator.export_to_json(results)

        import json
        data = json.loads(json_output)
        assert data["total"] == 1
        assert len(data["results"]) == 1

    def test_export_to_csv(self):
        generator = BatchReportGenerator()

        results = [
            AnalysisResult(
                defect_type="line_defect",
                korean_name="라인 결함",
                location="상단",
                severity="critical",
                reasoning="",
                root_causes=[{"cause": "Driver IC"}],
                recommended_actions=[{"action": "점검"}],
                related_processes=[],
                confidence=0.95,
                raw_response="",
            ),
        ]

        csv_output = generator.export_to_csv(results)

        assert "defect_type" in csv_output
        assert "line_defect" in csv_output
        assert "라인 결함" in csv_output
