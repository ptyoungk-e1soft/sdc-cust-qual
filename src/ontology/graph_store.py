"""Neo4j 그래프 저장소"""

import os
from contextlib import contextmanager
from typing import Any, Generator

from neo4j import GraphDatabase, Driver, Session

from .schema import (
    Defect,
    RootCause,
    Process,
    Equipment,
    Action,
    CausedByRelation,
    OccursInRelation,
    RequiresRelation,
)


class GraphStore:
    """Neo4j 기반 그래프 저장소"""

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        user: str = "neo4j",
        password: str | None = None,
        database: str = "neo4j",
    ):
        self.uri = uri
        self.user = user
        self.password = password or os.getenv("NEO4J_PASSWORD", "")
        self.database = database
        self._driver: Driver | None = None

    def connect(self) -> None:
        """데이터베이스 연결"""
        self._driver = GraphDatabase.driver(
            self.uri,
            auth=(self.user, self.password),
        )
        # 연결 테스트
        self._driver.verify_connectivity()

    def close(self) -> None:
        """연결 종료"""
        if self._driver:
            self._driver.close()
            self._driver = None

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        """세션 컨텍스트 매니저"""
        if not self._driver:
            self.connect()
        session = self._driver.session(database=self.database)
        try:
            yield session
        finally:
            session.close()

    def init_schema(self) -> None:
        """스키마 초기화 (인덱스 및 제약조건 생성)"""
        with self.session() as session:
            # 노드 유형별 유니크 제약조건
            constraints = [
                "CREATE CONSTRAINT defect_id IF NOT EXISTS FOR (d:Defect) REQUIRE d.defect_id IS UNIQUE",
                "CREATE CONSTRAINT cause_id IF NOT EXISTS FOR (c:RootCause) REQUIRE c.cause_id IS UNIQUE",
                "CREATE CONSTRAINT process_id IF NOT EXISTS FOR (p:Process) REQUIRE p.process_id IS UNIQUE",
                "CREATE CONSTRAINT equipment_id IF NOT EXISTS FOR (e:Equipment) REQUIRE e.equipment_id IS UNIQUE",
                "CREATE CONSTRAINT action_id IF NOT EXISTS FOR (a:Action) REQUIRE a.action_id IS UNIQUE",
            ]

            # 검색용 인덱스
            indexes = [
                "CREATE INDEX defect_type_idx IF NOT EXISTS FOR (d:Defect) ON (d.defect_type)",
                "CREATE INDEX cause_category_idx IF NOT EXISTS FOR (c:RootCause) ON (c.category)",
                "CREATE INDEX process_name_idx IF NOT EXISTS FOR (p:Process) ON (p.process_name)",
            ]

            for query in constraints + indexes:
                session.run(query)

    def clear_all(self) -> None:
        """모든 데이터 삭제"""
        with self.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    # 노드 CRUD 연산
    def create_defect(self, defect: Defect) -> None:
        """결함 노드 생성"""
        query = """
        MERGE (d:Defect {defect_id: $defect_id})
        SET d.defect_type = $defect_type,
            d.korean_name = $korean_name,
            d.description = $description,
            d.severity_levels = $severity_levels,
            d.visual_characteristics = $visual_characteristics
        """
        with self.session() as session:
            session.run(query, **defect.to_dict())

    def create_root_cause(self, cause: RootCause) -> None:
        """근본 원인 노드 생성"""
        query = """
        MERGE (c:RootCause {cause_id: $cause_id})
        SET c.cause_type = $cause_type,
            c.korean_name = $korean_name,
            c.description = $description,
            c.category = $category
        """
        with self.session() as session:
            session.run(query, **cause.to_dict())

    def create_process(self, process: Process) -> None:
        """공정 노드 생성"""
        query = """
        MERGE (p:Process {process_id: $process_id})
        SET p.process_name = $process_name,
            p.korean_name = $korean_name,
            p.sequence = $sequence,
            p.equipment_types = $equipment_types
        """
        with self.session() as session:
            session.run(query, **process.to_dict())

    def create_equipment(self, equipment: Equipment) -> None:
        """장비 노드 생성"""
        query = """
        MERGE (e:Equipment {equipment_id: $equipment_id})
        SET e.equipment_type = $equipment_type,
            e.korean_name = $korean_name,
            e.manufacturer = $manufacturer
        """
        with self.session() as session:
            session.run(query, **equipment.to_dict())

    def create_action(self, action: Action) -> None:
        """권장 조치 노드 생성"""
        query = """
        MERGE (a:Action {action_id: $action_id})
        SET a.action_type = $action_type,
            a.korean_name = $korean_name,
            a.description = $description,
            a.priority = $priority
        """
        with self.session() as session:
            session.run(query, **action.to_dict())

    # 관계 생성
    def create_caused_by(self, relation: CausedByRelation) -> None:
        """CAUSED_BY 관계 생성"""
        query = """
        MATCH (d:Defect {defect_id: $defect_id})
        MATCH (c:RootCause {cause_id: $cause_id})
        MERGE (d)-[r:CAUSED_BY]->(c)
        SET r.probability = $probability,
            r.evidence = $evidence
        """
        with self.session() as session:
            session.run(
                query,
                defect_id=relation.defect_id,
                cause_id=relation.cause_id,
                probability=relation.probability,
                evidence=relation.evidence,
            )

    def create_occurs_in(self, relation: OccursInRelation) -> None:
        """OCCURS_IN 관계 생성"""
        query = """
        MATCH (d:Defect {defect_id: $defect_id})
        MATCH (p:Process {process_id: $process_id})
        MERGE (d)-[r:OCCURS_IN]->(p)
        SET r.frequency = $frequency
        """
        with self.session() as session:
            session.run(
                query,
                defect_id=relation.defect_id,
                process_id=relation.process_id,
                frequency=relation.frequency,
            )

    def create_requires(self, relation: RequiresRelation) -> None:
        """REQUIRES 관계 생성"""
        query = """
        MATCH (c:RootCause {cause_id: $cause_id})
        MATCH (a:Action {action_id: $action_id})
        MERGE (c)-[r:REQUIRES]->(a)
        SET r.effectiveness = $effectiveness
        """
        with self.session() as session:
            session.run(
                query,
                cause_id=relation.cause_id,
                action_id=relation.action_id,
                effectiveness=relation.effectiveness,
            )

    # 쿼리 메서드
    def find_root_causes(
        self,
        defect_type: str,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """결함 유형에 대한 근본 원인 조회"""
        query = """
        MATCH (d:Defect {defect_type: $defect_type})-[r:CAUSED_BY]->(rc:RootCause)
        RETURN rc.korean_name AS cause,
               rc.category AS category,
               rc.description AS description,
               r.probability AS probability,
               r.evidence AS evidence
        ORDER BY r.probability DESC
        LIMIT $limit
        """
        with self.session() as session:
            result = session.run(query, defect_type=defect_type, limit=limit)
            return [dict(record) for record in result]

    def find_recommended_actions(
        self,
        defect_type: str,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """결함 유형에 대한 권장 조치 조회"""
        query = """
        MATCH (d:Defect {defect_type: $defect_type})-[:CAUSED_BY]->(rc:RootCause)-[r:REQUIRES]->(a:Action)
        RETURN a.korean_name AS action,
               a.description AS description,
               a.priority AS priority,
               r.effectiveness AS effectiveness,
               rc.korean_name AS for_cause
        ORDER BY r.effectiveness DESC
        LIMIT $limit
        """
        with self.session() as session:
            result = session.run(query, defect_type=defect_type, limit=limit)
            return [dict(record) for record in result]

    def find_related_processes(self, defect_type: str) -> list[dict[str, Any]]:
        """결함이 발생하는 공정 조회"""
        query = """
        MATCH (d:Defect {defect_type: $defect_type})-[r:OCCURS_IN]->(p:Process)
        RETURN p.korean_name AS process,
               p.process_name AS process_name,
               r.frequency AS frequency,
               p.sequence AS sequence
        ORDER BY p.sequence
        """
        with self.session() as session:
            result = session.run(query, defect_type=defect_type)
            return [dict(record) for record in result]

    def get_defect_analysis(self, defect_type: str) -> dict[str, Any]:
        """결함 유형에 대한 종합 분석"""
        return {
            "defect_type": defect_type,
            "root_causes": self.find_root_causes(defect_type),
            "recommended_actions": self.find_recommended_actions(defect_type),
            "related_processes": self.find_related_processes(defect_type),
        }

    def execute_query(self, query: str, params: dict | None = None) -> list[dict[str, Any]]:
        """커스텀 Cypher 쿼리 실행"""
        with self.session() as session:
            result = session.run(query, **(params or {}))
            return [dict(record) for record in result]


class InMemoryGraphStore:
    """테스트 및 경량 환경을 위한 인메모리 그래프 저장소"""

    def __init__(self):
        self.defects: dict[str, Defect] = {}
        self.root_causes: dict[str, RootCause] = {}
        self.processes: dict[str, Process] = {}
        self.equipment: dict[str, Equipment] = {}
        self.actions: dict[str, Action] = {}

        self.caused_by: list[CausedByRelation] = []
        self.occurs_in: list[OccursInRelation] = []
        self.requires: list[RequiresRelation] = []

    def connect(self) -> None:
        pass

    def close(self) -> None:
        pass

    def init_schema(self) -> None:
        pass

    def clear_all(self) -> None:
        self.defects.clear()
        self.root_causes.clear()
        self.processes.clear()
        self.equipment.clear()
        self.actions.clear()
        self.caused_by.clear()
        self.occurs_in.clear()
        self.requires.clear()

    def create_defect(self, defect: Defect) -> None:
        self.defects[defect.defect_id] = defect

    def create_root_cause(self, cause: RootCause) -> None:
        self.root_causes[cause.cause_id] = cause

    def create_process(self, process: Process) -> None:
        self.processes[process.process_id] = process

    def create_equipment(self, equipment: Equipment) -> None:
        self.equipment[equipment.equipment_id] = equipment

    def create_action(self, action: Action) -> None:
        self.actions[action.action_id] = action

    def create_caused_by(self, relation: CausedByRelation) -> None:
        self.caused_by.append(relation)

    def create_occurs_in(self, relation: OccursInRelation) -> None:
        self.occurs_in.append(relation)

    def create_requires(self, relation: RequiresRelation) -> None:
        self.requires.append(relation)

    def find_root_causes(self, defect_type: str, limit: int = 5) -> list[dict[str, Any]]:
        results = []
        defect_ids = [d.defect_id for d in self.defects.values() if d.defect_type.value == defect_type]

        for rel in self.caused_by:
            if rel.defect_id in defect_ids:
                cause = self.root_causes.get(rel.cause_id)
                if cause:
                    results.append(
                        {
                            "cause": cause.korean_name,
                            "category": cause.category.value,
                            "description": cause.description,
                            "probability": rel.probability,
                            "evidence": rel.evidence,
                        }
                    )

        results.sort(key=lambda x: x["probability"], reverse=True)
        return results[:limit]

    def find_recommended_actions(self, defect_type: str, limit: int = 5) -> list[dict[str, Any]]:
        root_causes = self.find_root_causes(defect_type, limit=100)
        cause_names = {r["cause"] for r in root_causes}

        cause_ids = [c.cause_id for c in self.root_causes.values() if c.korean_name in cause_names]

        results = []
        for rel in self.requires:
            if rel.cause_id in cause_ids:
                action = self.actions.get(rel.action_id)
                cause = self.root_causes.get(rel.cause_id)
                if action and cause:
                    results.append(
                        {
                            "action": action.korean_name,
                            "description": action.description,
                            "priority": action.priority.value,
                            "effectiveness": rel.effectiveness,
                            "for_cause": cause.korean_name,
                        }
                    )

        results.sort(key=lambda x: x["effectiveness"], reverse=True)
        return results[:limit]

    def find_related_processes(self, defect_type: str) -> list[dict[str, Any]]:
        results = []
        defect_ids = [d.defect_id for d in self.defects.values() if d.defect_type.value == defect_type]

        for rel in self.occurs_in:
            if rel.defect_id in defect_ids:
                process = self.processes.get(rel.process_id)
                if process:
                    results.append(
                        {
                            "process": process.korean_name,
                            "process_name": process.process_name,
                            "frequency": rel.frequency,
                            "sequence": process.sequence,
                        }
                    )

        results.sort(key=lambda x: x["sequence"])
        return results

    def get_defect_analysis(self, defect_type: str) -> dict[str, Any]:
        return {
            "defect_type": defect_type,
            "root_causes": self.find_root_causes(defect_type),
            "recommended_actions": self.find_recommended_actions(defect_type),
            "related_processes": self.find_related_processes(defect_type),
        }
