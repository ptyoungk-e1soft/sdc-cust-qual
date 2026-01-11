"""
빅데이터 스키마 정의
개발단계, 제조현장, MES 실적 데이터 스키마
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum


class DevelopmentPhase(Enum):
    """개발 단계"""
    EVT = "EVT"  # Engineering Validation Test
    DVT = "DVT"  # Design Validation Test
    PVT = "PVT"  # Production Validation Test
    MP = "MP"    # Mass Production


class DefectSeverity(Enum):
    """결함 심각도"""
    CRITICAL = "CRITICAL"
    MAJOR = "MAJOR"
    MINOR = "MINOR"
    COSMETIC = "COSMETIC"


class InspectionResult(Enum):
    """검사 결과"""
    PASS = "PASS"
    FAIL = "FAIL"
    REWORK = "REWORK"
    SCRAP = "SCRAP"


@dataclass
class DevelopmentData:
    """개발단계 데이터"""
    project_id: str
    phase: str
    sample_id: str
    test_date: datetime
    test_type: str
    test_result: str
    measurement_values: Dict[str, float]
    engineer: str
    notes: str = ""
    related_issues: List[str] = field(default_factory=list)


@dataclass
class EquipmentData:
    """설비 데이터"""
    equipment_id: str
    equipment_name: str
    equipment_type: str
    line_id: str
    vendor: str
    install_date: datetime
    last_pm_date: datetime
    status: str
    utilization: float
    parameters: Dict[str, float] = field(default_factory=dict)


@dataclass
class MaterialData:
    """자재 데이터"""
    material_id: str
    material_name: str
    material_type: str
    lot_no: str
    vendor: str
    receive_date: datetime
    expire_date: datetime
    quantity: float
    unit: str
    quality_grade: str
    storage_location: str
    specifications: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InspectionData:
    """검사 데이터"""
    inspection_id: str
    lot_id: str
    cell_id: str
    inspection_type: str
    inspection_step: str
    inspector_id: str
    inspection_time: datetime
    result: str
    defect_codes: List[str]
    measurements: Dict[str, float]
    images: List[str] = field(default_factory=list)


@dataclass
class QualityData:
    """양/불량 데이터"""
    lot_id: str
    cell_id: str
    process_step: str
    inspection_result: str
    defect_type: str
    defect_location: str
    defect_size: float
    severity: str
    root_cause: str
    corrective_action: str
    timestamp: datetime
    equipment_id: str = ""
    operator_id: str = ""


@dataclass
class ManufacturingData:
    """제조현장 통합 데이터"""
    lot_id: str
    wafer_id: str
    cell_id: str
    process_step: str
    equipment: EquipmentData
    materials: List[MaterialData]
    inspection: InspectionData
    quality: QualityData
    process_time: datetime
    cycle_time: float
    yield_rate: float


@dataclass
class MESData:
    """MES 실적 데이터"""
    work_order_id: str
    lot_id: str
    product_id: str
    product_name: str
    customer: str
    plan_qty: int
    actual_qty: int
    good_qty: int
    ng_qty: int
    yield_rate: float
    start_time: datetime
    end_time: datetime
    line_id: str
    shift: str
    operator_id: str
    process_route: List[str]
    status: str
    remarks: str = ""


@dataclass
class TracebilityData:
    """이력추적 데이터"""
    trace_id: str
    cell_id: str
    lot_id: str
    timestamp: datetime
    event_type: str
    event_detail: str
    equipment_id: str
    material_lots: List[str]
    parameters: Dict[str, float]
    quality_result: str
    linked_defects: List[str] = field(default_factory=list)
