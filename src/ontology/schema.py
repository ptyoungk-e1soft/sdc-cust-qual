"""온톨로지 스키마 정의

디스플레이 결함 분석을 위한 지식 그래프 스키마
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class DefectType(str, Enum):
    """결함 유형"""

    DEAD_PIXEL = "dead_pixel"
    BRIGHT_SPOT = "bright_spot"
    LINE_DEFECT = "line_defect"
    MURA = "mura"
    SCRATCH = "scratch"
    PARTICLE = "particle"
    CUSTOM = "custom"

    @property
    def korean_name(self) -> str:
        names = {
            "dead_pixel": "데드 픽셀",
            "bright_spot": "휘점 결함",
            "line_defect": "라인 결함",
            "mura": "무라 (얼룩)",
            "scratch": "스크래치",
            "particle": "이물질",
            "custom": "기타 결함",
        }
        return names.get(self.value, self.value)


class SeverityLevel(str, Enum):
    """심각도 수준"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def korean_name(self) -> str:
        names = {
            "low": "경미",
            "medium": "보통",
            "high": "심각",
            "critical": "치명적",
        }
        return names.get(self.value, self.value)


class CauseCategory(str, Enum):
    """원인 분류"""

    EQUIPMENT = "equipment"
    PROCESS = "process"
    MATERIAL = "material"
    ENVIRONMENT = "environment"
    HUMAN = "human"

    @property
    def korean_name(self) -> str:
        names = {
            "equipment": "장비",
            "process": "공정",
            "material": "재료",
            "environment": "환경",
            "human": "인적",
        }
        return names.get(self.value, self.value)


class ActionPriority(str, Enum):
    """조치 우선순위"""

    IMMEDIATE = "immediate"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class Defect:
    """결함 노드"""

    defect_id: str
    defect_type: DefectType
    korean_name: str
    description: str = ""
    severity_levels: list[SeverityLevel] = field(default_factory=list)
    visual_characteristics: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "defect_id": self.defect_id,
            "defect_type": self.defect_type.value,
            "korean_name": self.korean_name,
            "description": self.description,
            "severity_levels": [s.value for s in self.severity_levels],
            "visual_characteristics": self.visual_characteristics,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Defect":
        return cls(
            defect_id=data["defect_id"],
            defect_type=DefectType(data["defect_type"]),
            korean_name=data["korean_name"],
            description=data.get("description", ""),
            severity_levels=[SeverityLevel(s) for s in data.get("severity_levels", [])],
            visual_characteristics=data.get("visual_characteristics", ""),
        )


@dataclass
class RootCause:
    """근본 원인 노드"""

    cause_id: str
    cause_type: str
    korean_name: str
    description: str = ""
    category: CauseCategory = CauseCategory.PROCESS

    def to_dict(self) -> dict[str, Any]:
        return {
            "cause_id": self.cause_id,
            "cause_type": self.cause_type,
            "korean_name": self.korean_name,
            "description": self.description,
            "category": self.category.value,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RootCause":
        return cls(
            cause_id=data["cause_id"],
            cause_type=data["cause_type"],
            korean_name=data["korean_name"],
            description=data.get("description", ""),
            category=CauseCategory(data.get("category", "process")),
        )


@dataclass
class Process:
    """공정 노드"""

    process_id: str
    process_name: str
    korean_name: str
    sequence: int = 0
    equipment_types: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "process_id": self.process_id,
            "process_name": self.process_name,
            "korean_name": self.korean_name,
            "sequence": self.sequence,
            "equipment_types": self.equipment_types,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Process":
        return cls(
            process_id=data["process_id"],
            process_name=data["process_name"],
            korean_name=data["korean_name"],
            sequence=data.get("sequence", 0),
            equipment_types=data.get("equipment_types", []),
        )


@dataclass
class Equipment:
    """장비 노드"""

    equipment_id: str
    equipment_type: str
    korean_name: str
    manufacturer: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "equipment_id": self.equipment_id,
            "equipment_type": self.equipment_type,
            "korean_name": self.korean_name,
            "manufacturer": self.manufacturer,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Equipment":
        return cls(
            equipment_id=data["equipment_id"],
            equipment_type=data["equipment_type"],
            korean_name=data["korean_name"],
            manufacturer=data.get("manufacturer", ""),
        )


@dataclass
class Action:
    """권장 조치 노드"""

    action_id: str
    action_type: str
    korean_name: str
    description: str = ""
    priority: ActionPriority = ActionPriority.MEDIUM

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "action_type": self.action_type,
            "korean_name": self.korean_name,
            "description": self.description,
            "priority": self.priority.value,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Action":
        return cls(
            action_id=data["action_id"],
            action_type=data["action_type"],
            korean_name=data["korean_name"],
            description=data.get("description", ""),
            priority=ActionPriority(data.get("priority", "medium")),
        )


@dataclass
class CausedByRelation:
    """CAUSED_BY 관계"""

    defect_id: str
    cause_id: str
    probability: float = 0.5
    evidence: str = ""


@dataclass
class OccursInRelation:
    """OCCURS_IN 관계"""

    defect_id: str
    process_id: str
    frequency: str = "occasional"


@dataclass
class RequiresRelation:
    """REQUIRES 관계"""

    cause_id: str
    action_id: str
    effectiveness: float = 0.5


# 기본 온톨로지 인스턴스 생성
def get_default_defects() -> list[Defect]:
    """기본 결함 유형 정의"""
    return [
        Defect(
            defect_id="DEF001",
            defect_type=DefectType.DEAD_PIXEL,
            korean_name="데드 픽셀",
            description="화면에 검은 점으로 나타나는 비활성 픽셀",
            severity_levels=[SeverityLevel.LOW, SeverityLevel.MEDIUM, SeverityLevel.HIGH],
            visual_characteristics="검은색 또는 어두운 점, 개별 또는 군집",
        ),
        Defect(
            defect_id="DEF002",
            defect_type=DefectType.BRIGHT_SPOT,
            korean_name="휘점 결함",
            description="화면에 밝은 점으로 나타나는 결함",
            severity_levels=[SeverityLevel.LOW, SeverityLevel.MEDIUM, SeverityLevel.HIGH],
            visual_characteristics="밝은 점, 흰색 또는 색상 점",
        ),
        Defect(
            defect_id="DEF003",
            defect_type=DefectType.LINE_DEFECT,
            korean_name="라인 결함",
            description="수평 또는 수직 방향의 선 형태 결함",
            severity_levels=[SeverityLevel.MEDIUM, SeverityLevel.HIGH, SeverityLevel.CRITICAL],
            visual_characteristics="수직/수평선, 단일 또는 다중",
        ),
        Defect(
            defect_id="DEF004",
            defect_type=DefectType.MURA,
            korean_name="무라 (얼룩)",
            description="불균일한 밝기 또는 색상 분포",
            severity_levels=[SeverityLevel.LOW, SeverityLevel.MEDIUM, SeverityLevel.HIGH],
            visual_characteristics="얼룩 패턴, 그라데이션 불균일",
        ),
        Defect(
            defect_id="DEF005",
            defect_type=DefectType.SCRATCH,
            korean_name="스크래치",
            description="표면 긁힘 결함",
            severity_levels=[SeverityLevel.LOW, SeverityLevel.MEDIUM, SeverityLevel.HIGH],
            visual_characteristics="선형 긁힘, 불규칙한 방향",
        ),
        Defect(
            defect_id="DEF006",
            defect_type=DefectType.PARTICLE,
            korean_name="이물질",
            description="제조 공정 중 혼입된 이물질",
            severity_levels=[
                SeverityLevel.LOW,
                SeverityLevel.MEDIUM,
                SeverityLevel.HIGH,
                SeverityLevel.CRITICAL,
            ],
            visual_characteristics="불규칙한 형태의 점 또는 덩어리",
        ),
    ]


def get_default_root_causes() -> list[RootCause]:
    """기본 근본 원인 정의"""
    return [
        RootCause(
            cause_id="RC001",
            cause_type="tft_manufacturing_defect",
            korean_name="TFT 제조 결함",
            description="TFT 어레이 제조 공정 중 발생한 결함",
            category=CauseCategory.PROCESS,
        ),
        RootCause(
            cause_id="RC002",
            cause_type="contamination",
            korean_name="오염",
            description="클린룸 환경 오염 또는 이물질 유입",
            category=CauseCategory.ENVIRONMENT,
        ),
        RootCause(
            cause_id="RC003",
            cause_type="alignment_error",
            korean_name="정렬 오류",
            description="층간 정렬 불량",
            category=CauseCategory.EQUIPMENT,
        ),
        RootCause(
            cause_id="RC004",
            cause_type="material_defect",
            korean_name="재료 결함",
            description="원재료 품질 문제",
            category=CauseCategory.MATERIAL,
        ),
        RootCause(
            cause_id="RC005",
            cause_type="handling_damage",
            korean_name="핸들링 손상",
            description="운반 또는 취급 중 발생한 손상",
            category=CauseCategory.HUMAN,
        ),
        RootCause(
            cause_id="RC006",
            cause_type="etching_defect",
            korean_name="식각 결함",
            description="에칭 공정 중 발생한 문제",
            category=CauseCategory.PROCESS,
        ),
        RootCause(
            cause_id="RC007",
            cause_type="deposition_defect",
            korean_name="증착 결함",
            description="박막 증착 공정 불량",
            category=CauseCategory.PROCESS,
        ),
        RootCause(
            cause_id="RC008",
            cause_type="driver_ic_defect",
            korean_name="Driver IC 불량",
            description="드라이버 IC 또는 연결 불량",
            category=CauseCategory.MATERIAL,
        ),
    ]


def get_default_actions() -> list[Action]:
    """기본 권장 조치 정의"""
    return [
        Action(
            action_id="ACT001",
            action_type="equipment_inspection",
            korean_name="장비 점검",
            description="관련 장비의 상태 점검 및 유지보수",
            priority=ActionPriority.HIGH,
        ),
        Action(
            action_id="ACT002",
            action_type="process_adjustment",
            korean_name="공정 조건 조정",
            description="공정 파라미터 재설정",
            priority=ActionPriority.MEDIUM,
        ),
        Action(
            action_id="ACT003",
            action_type="material_change",
            korean_name="재료 교체",
            description="원재료 로트 변경 또는 공급업체 검토",
            priority=ActionPriority.MEDIUM,
        ),
        Action(
            action_id="ACT004",
            action_type="environment_check",
            korean_name="환경 점검",
            description="클린룸 환경 점검 (온습도, 파티클 등)",
            priority=ActionPriority.HIGH,
        ),
        Action(
            action_id="ACT005",
            action_type="operator_training",
            korean_name="작업자 교육",
            description="핸들링 절차 재교육",
            priority=ActionPriority.LOW,
        ),
    ]
