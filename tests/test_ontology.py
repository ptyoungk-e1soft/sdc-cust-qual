"""온톨로지 모듈 테스트"""

import pytest

from src.ontology.schema import (
    Defect,
    RootCause,
    Action,
    DefectType,
    CauseCategory,
    ActionPriority,
    SeverityLevel,
    get_default_defects,
    get_default_root_causes,
    get_default_actions,
)
from src.ontology.graph_store import InMemoryGraphStore
from src.ontology.knowledge_base import KnowledgeBase
from src.ontology.reasoning import RootCauseReasoner, DefectEvidence


class TestSchema:
    """스키마 테스트"""

    def test_defect_type_enum(self):
        assert DefectType.DEAD_PIXEL.value == "dead_pixel"
        assert DefectType.DEAD_PIXEL.korean_name == "데드 픽셀"

    def test_severity_level_enum(self):
        assert SeverityLevel.HIGH.korean_name == "심각"

    def test_defect_to_dict(self):
        defect = Defect(
            defect_id="DEF001",
            defect_type=DefectType.DEAD_PIXEL,
            korean_name="데드 픽셀",
            description="테스트",
        )
        data = defect.to_dict()
        assert data["defect_id"] == "DEF001"
        assert data["defect_type"] == "dead_pixel"

    def test_default_defects(self):
        defects = get_default_defects()
        assert len(defects) >= 6
        assert any(d.defect_type == DefectType.DEAD_PIXEL for d in defects)


class TestInMemoryGraphStore:
    """인메모리 그래프 저장소 테스트"""

    def test_create_defect(self):
        store = InMemoryGraphStore()
        defect = Defect(
            defect_id="DEF001",
            defect_type=DefectType.DEAD_PIXEL,
            korean_name="데드 픽셀",
        )
        store.create_defect(defect)
        assert "DEF001" in store.defects

    def test_create_root_cause(self):
        store = InMemoryGraphStore()
        cause = RootCause(
            cause_id="RC001",
            cause_type="test_cause",
            korean_name="테스트 원인",
            category=CauseCategory.PROCESS,
        )
        store.create_root_cause(cause)
        assert "RC001" in store.root_causes


class TestKnowledgeBase:
    """지식 베이스 테스트"""

    def test_init_with_defaults(self):
        kb = KnowledgeBase(use_neo4j=False)
        kb.load_defaults()

        # 결함 분석
        analysis = kb.analyze_defect("dead_pixel")
        assert "root_causes" in analysis
        assert len(analysis["root_causes"]) > 0

    def test_get_root_causes(self):
        kb = KnowledgeBase(use_neo4j=False)
        kb.load_defaults()

        causes = kb.get_root_causes("dead_pixel")
        assert len(causes) > 0

    def test_get_recommended_actions(self):
        kb = KnowledgeBase(use_neo4j=False)
        kb.load_defaults()

        actions = kb.get_recommended_actions("dead_pixel")
        assert len(actions) > 0


class TestRootCauseReasoner:
    """근본원인 추론 테스트"""

    def test_reason(self):
        kb = KnowledgeBase(use_neo4j=False)
        kb.load_defaults()

        reasoner = RootCauseReasoner(kb)

        evidence = DefectEvidence(
            defect_type="dead_pixel",
            location="중앙",
            severity="medium",
        )

        result = reasoner.reason(evidence)

        assert result.defect_type == "dead_pixel"
        assert result.confidence > 0
        assert len(result.root_causes) > 0
        assert len(result.reasoning_chain) > 0

    def test_format_reasoning_output(self):
        kb = KnowledgeBase(use_neo4j=False)
        kb.load_defaults()

        reasoner = RootCauseReasoner(kb)

        evidence = DefectEvidence(
            defect_type="dead_pixel",
            location="좌측 상단",
            severity="high",
        )

        result = reasoner.reason(evidence)
        output = reasoner.format_reasoning_output(result)

        assert "<think>" in output
        assert "</think>" in output
        assert "<answer>" in output
        assert "</answer>" in output
