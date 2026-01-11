"""지식 베이스 관리"""

import os
from pathlib import Path
from typing import Any

import yaml

from .schema import (
    Defect,
    RootCause,
    Process,
    Equipment,
    Action,
    DefectType,
    CauseCategory,
    ActionPriority,
    SeverityLevel,
    CausedByRelation,
    OccursInRelation,
    RequiresRelation,
    get_default_defects,
    get_default_root_causes,
    get_default_actions,
)
from .graph_store import GraphStore, InMemoryGraphStore


class KnowledgeBase:
    """디스플레이 결함 분석 지식 베이스"""

    def __init__(
        self,
        config_path: str | Path | None = None,
        use_neo4j: bool = True,
    ):
        self.config_path = Path(config_path) if config_path else None
        self.use_neo4j = use_neo4j

        if use_neo4j:
            self.store = self._init_neo4j_store()
        else:
            self.store = InMemoryGraphStore()

    def _init_neo4j_store(self) -> GraphStore:
        """Neo4j 저장소 초기화"""
        if self.config_path and self.config_path.exists():
            with open(self.config_path, encoding="utf-8") as f:
                config = yaml.safe_load(f)

            db_config = config.get("database", {})
            uri = db_config.get("uri", "bolt://localhost:7687")
            user = db_config.get("user", "neo4j")
            password = db_config.get("password", "").replace("${NEO4J_PASSWORD}", os.getenv("NEO4J_PASSWORD", ""))
            database = db_config.get("database", "neo4j")

            return GraphStore(uri=uri, user=user, password=password, database=database)

        return GraphStore()

    def connect(self) -> None:
        """저장소 연결"""
        self.store.connect()
        self.store.init_schema()

    def close(self) -> None:
        """연결 종료"""
        self.store.close()

    def load_from_config(self, config_path: str | Path | None = None) -> None:
        """설정 파일에서 온톨로지 로드"""
        config_path = Path(config_path or self.config_path)
        if not config_path or not config_path.exists():
            raise ValueError(f"Config file not found: {config_path}")

        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        self._load_nodes(config)
        self._load_relationships(config)

    def _load_nodes(self, config: dict) -> None:
        """노드 로드"""
        node_types = config.get("node_types", {})

        # Defect 노드
        defect_config = node_types.get("Defect", {})
        for instance in defect_config.get("instances", []):
            defect = Defect(
                defect_id=instance["defect_id"],
                defect_type=DefectType(instance["defect_type"]),
                korean_name=instance["korean_name"],
                description=instance.get("description", ""),
                severity_levels=[SeverityLevel(s) for s in instance.get("severity_levels", [])],
                visual_characteristics=instance.get("visual_characteristics", ""),
            )
            self.store.create_defect(defect)

        # RootCause 노드
        cause_config = node_types.get("RootCause", {})
        for instance in cause_config.get("instances", []):
            cause = RootCause(
                cause_id=instance["cause_id"],
                cause_type=instance["cause_type"],
                korean_name=instance["korean_name"],
                description=instance.get("description", ""),
                category=CauseCategory(instance.get("category", "process")),
            )
            self.store.create_root_cause(cause)

        # Process 노드
        process_config = node_types.get("Process", {})
        for instance in process_config.get("instances", []):
            process = Process(
                process_id=instance["process_id"],
                process_name=instance["process_name"],
                korean_name=instance["korean_name"],
                sequence=instance.get("sequence", 0),
                equipment_types=instance.get("equipment_types", []),
            )
            self.store.create_process(process)

        # Equipment 노드
        equipment_config = node_types.get("Equipment", {})
        for instance in equipment_config.get("instances", []):
            equipment = Equipment(
                equipment_id=instance["equipment_id"],
                equipment_type=instance["equipment_type"],
                korean_name=instance["korean_name"],
                manufacturer=instance.get("manufacturer", ""),
            )
            self.store.create_equipment(equipment)

        # Action 노드
        action_config = node_types.get("Action", {})
        for instance in action_config.get("instances", []):
            action = Action(
                action_id=instance["action_id"],
                action_type=instance["action_type"],
                korean_name=instance["korean_name"],
                description=instance.get("description", ""),
                priority=ActionPriority(instance.get("priority", "medium")),
            )
            self.store.create_action(action)

    def _load_relationships(self, config: dict) -> None:
        """관계 로드"""
        rel_types = config.get("relationship_types", {})

        # CAUSED_BY 관계
        caused_by_config = rel_types.get("CAUSED_BY", {})
        for instance in caused_by_config.get("instances", []):
            relation = CausedByRelation(
                defect_id=instance["from"],
                cause_id=instance["to"],
                probability=instance.get("probability", 0.5),
                evidence=instance.get("evidence", ""),
            )
            self.store.create_caused_by(relation)

        # OCCURS_IN 관계
        occurs_in_config = rel_types.get("OCCURS_IN", {})
        for instance in occurs_in_config.get("instances", []):
            relation = OccursInRelation(
                defect_id=instance["from"],
                process_id=instance["to"],
                frequency=instance.get("frequency", "occasional"),
            )
            self.store.create_occurs_in(relation)

        # REQUIRES 관계
        requires_config = rel_types.get("REQUIRES", {})
        for instance in requires_config.get("instances", []):
            relation = RequiresRelation(
                cause_id=instance["from"],
                action_id=instance["to"],
                effectiveness=instance.get("effectiveness", 0.5),
            )
            self.store.create_requires(relation)

    def load_defaults(self) -> None:
        """기본 온톨로지 로드"""
        # 기본 노드 생성
        for defect in get_default_defects():
            self.store.create_defect(defect)

        for cause in get_default_root_causes():
            self.store.create_root_cause(cause)

        for action in get_default_actions():
            self.store.create_action(action)

        # 기본 관계 생성
        default_relations = [
            CausedByRelation("DEF001", "RC001", 0.7, "TFT 트랜지스터 제조 결함으로 인한 픽셀 비활성화"),
            CausedByRelation("DEF001", "RC006", 0.2, "과도한 에칭으로 인한 트랜지스터 손상"),
            CausedByRelation("DEF002", "RC007", 0.6, "증착 불균일로 인한 전하 누적"),
            CausedByRelation("DEF003", "RC008", 0.8, "Driver IC 불량으로 인한 라인 구동 실패"),
            CausedByRelation("DEF003", "RC003", 0.15, "정렬 오류로 인한 라인 단락"),
            CausedByRelation("DEF004", "RC007", 0.7, "증착 두께 불균일"),
            CausedByRelation("DEF005", "RC005", 0.9, "부주의한 취급으로 인한 표면 손상"),
            CausedByRelation("DEF006", "RC002", 0.85, "클린룸 오염으로 인한 이물질 혼입"),
        ]

        for rel in default_relations:
            self.store.create_caused_by(rel)

        # 조치 관계
        action_relations = [
            RequiresRelation("RC001", "ACT001", 0.8),
            RequiresRelation("RC002", "ACT004", 0.9),
            RequiresRelation("RC003", "ACT001", 0.85),
            RequiresRelation("RC004", "ACT003", 0.75),
            RequiresRelation("RC005", "ACT005", 0.7),
            RequiresRelation("RC006", "ACT002", 0.8),
            RequiresRelation("RC007", "ACT002", 0.8),
            RequiresRelation("RC008", "ACT003", 0.85),
        ]

        for rel in action_relations:
            self.store.create_requires(rel)

    def analyze_defect(self, defect_type: str) -> dict[str, Any]:
        """결함 유형 분석"""
        return self.store.get_defect_analysis(defect_type)

    def get_root_causes(self, defect_type: str, limit: int = 5) -> list[dict[str, Any]]:
        """근본 원인 조회"""
        return self.store.find_root_causes(defect_type, limit=limit)

    def get_recommended_actions(self, defect_type: str, limit: int = 5) -> list[dict[str, Any]]:
        """권장 조치 조회"""
        return self.store.find_recommended_actions(defect_type, limit=limit)

    def get_related_processes(self, defect_type: str) -> list[dict[str, Any]]:
        """관련 공정 조회"""
        return self.store.find_related_processes(defect_type)

    def format_analysis_report(self, defect_type: str) -> str:
        """분석 보고서 포맷팅"""
        analysis = self.analyze_defect(defect_type)

        report_lines = [
            f"## 결함 분석 보고서: {defect_type}",
            "",
            "### 추정 원인 (확률순)",
        ]

        for i, cause in enumerate(analysis.get("root_causes", []), 1):
            prob = cause.get("probability", 0) * 100
            report_lines.append(
                f"{i}. **{cause.get('cause')}** ({prob:.0f}%)"
            )
            if cause.get("evidence"):
                report_lines.append(f"   - 근거: {cause.get('evidence')}")

        report_lines.extend(["", "### 권장 조치"])

        for i, action in enumerate(analysis.get("recommended_actions", []), 1):
            eff = action.get("effectiveness", 0) * 100
            report_lines.append(
                f"{i}. **{action.get('action')}** (효과성: {eff:.0f}%)"
            )
            report_lines.append(f"   - {action.get('description')}")
            report_lines.append(f"   - 우선순위: {action.get('priority')}")
            report_lines.append(f"   - 대상 원인: {action.get('for_cause')}")

        report_lines.extend(["", "### 관련 공정"])

        for process in analysis.get("related_processes", []):
            report_lines.append(
                f"- {process.get('process')} (빈도: {process.get('frequency')})"
            )

        return "\n".join(report_lines)
