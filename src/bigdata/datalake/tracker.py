"""
제품 이력 추적기
불량 셀의 전체 제조 이력 추적
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ProductTrace:
    """제품 추적 결과"""
    cell_id: str
    lot_id: str
    trace_time: datetime
    process_steps: List[Dict[str, Any]]
    equipment_list: List[str]
    recipe_list: List[str]
    anomalies: List[Dict[str, Any]]


class ProductHistoryTracker:
    """
    제품 이력 추적기

    불량 셀의 전체 제조 공정 이력을 추적하고
    관련 설비, 레시피, 이상 징후를 분석
    """

    def __init__(self, greenplum_connector=None, oracle_connector=None):
        self.gp = greenplum_connector
        self.oracle = oracle_connector

    def trace_product(self, cell_id: str, lot_id: Optional[str] = None) -> ProductTrace:
        """
        제품 이력 추적

        Args:
            cell_id: 셀 ID
            lot_id: 로트 ID (선택)

        Returns:
            추적 결과
        """
        logger.info(f"제품 이력 추적 시작: {cell_id}")

        # 공정 단계 조회
        process_steps = self._get_process_steps(cell_id, lot_id)

        # 설비 목록 추출
        equipment_list = list(set([
            step.get("equipment_id") for step in process_steps
            if step.get("equipment_id")
        ]))

        # 레시피 목록 추출
        recipe_list = list(set([
            step.get("recipe_id") for step in process_steps
            if step.get("recipe_id")
        ]))

        # 이상 징후 분석
        anomalies = self._detect_anomalies(process_steps)

        return ProductTrace(
            cell_id=cell_id,
            lot_id=lot_id or (process_steps[0].get("lot_id") if process_steps else ""),
            trace_time=datetime.now(),
            process_steps=process_steps,
            equipment_list=equipment_list,
            recipe_list=recipe_list,
            anomalies=anomalies
        )

    def _get_process_steps(self, cell_id: str, lot_id: Optional[str]) -> List[Dict[str, Any]]:
        """공정 단계 조회"""
        if self.gp:
            sql = f"""
            SELECT
                lot_id, wafer_id, cell_id, process_step,
                equipment_id, recipe_id, process_time,
                param_1, param_2, param_3
            FROM qms.product_history
            WHERE cell_id = '{cell_id}'
            ORDER BY process_time
            """
            return self.gp.query(sql)

        # 시뮬레이션 데이터
        return [
            {
                "lot_id": lot_id or "L20250101001",
                "wafer_id": "W001",
                "cell_id": cell_id,
                "process_step": "TFT_ARRAY",
                "equipment_id": "EQ001",
                "recipe_id": "RCP001",
                "process_time": "2025-01-01 10:00:00",
                "param_1": 350.5,
                "param_2": 25.0,
                "param_3": 0.05
            },
            {
                "lot_id": lot_id or "L20250101001",
                "wafer_id": "W001",
                "cell_id": cell_id,
                "process_step": "CF_ARRAY",
                "equipment_id": "EQ002",
                "recipe_id": "RCP002",
                "process_time": "2025-01-01 14:00:00",
                "param_1": 280.0,
                "param_2": 22.5,
                "param_3": 0.03
            },
            {
                "lot_id": lot_id or "L20250101001",
                "wafer_id": "W001",
                "cell_id": cell_id,
                "process_step": "CELL_ASSEMBLY",
                "equipment_id": "EQ003",
                "recipe_id": "RCP003",
                "process_time": "2025-01-02 09:00:00",
                "param_1": 25.0,
                "param_2": 50.0,
                "param_3": 0.01
            }
        ]

    def _detect_anomalies(self, process_steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """이상 징후 탐지"""
        anomalies = []

        # 간단한 임계값 기반 탐지 (시뮬레이션)
        for step in process_steps:
            if step.get("param_1", 0) > 340:
                anomalies.append({
                    "process_step": step.get("process_step"),
                    "equipment_id": step.get("equipment_id"),
                    "parameter": "param_1",
                    "value": step.get("param_1"),
                    "threshold": 340,
                    "type": "HIGH_TEMPERATURE"
                })

        return anomalies

    def get_related_lots(self, cell_id: str, time_window_hours: int = 24) -> List[str]:
        """
        관련 로트 조회 (동일 설비에서 처리된 로트)
        """
        if self.gp:
            sql = f"""
            SELECT DISTINCT lot_id
            FROM qms.product_history
            WHERE equipment_id IN (
                SELECT equipment_id FROM qms.product_history
                WHERE cell_id = '{cell_id}'
            )
            AND process_time BETWEEN (
                SELECT MIN(process_time) - INTERVAL '{time_window_hours} hours'
                FROM qms.product_history WHERE cell_id = '{cell_id}'
            ) AND (
                SELECT MAX(process_time) + INTERVAL '{time_window_hours} hours'
                FROM qms.product_history WHERE cell_id = '{cell_id}'
            )
            """
            results = self.gp.query(sql)
            return [r.get("lot_id") for r in results if r.get("lot_id")]

        # 시뮬레이션
        return ["L20250101001", "L20250101002", "L20250101003"]
