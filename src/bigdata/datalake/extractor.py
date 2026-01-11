"""
불량 데이터 추출기
고객 품질 불량 발생 시 관련 모든 데이터 추출
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class DefectCase:
    """고객 품질 불량 케이스"""
    case_id: str
    cell_id: str
    defect_type: str
    defect_date: datetime
    customer: str
    severity: str = "MEDIUM"
    description: str = ""
    lot_ids: List[str] = field(default_factory=list)


@dataclass
class ExtractedData:
    """추출된 데이터"""
    case_id: str
    cell_id: str
    extraction_time: datetime
    product_history: List[Dict[str, Any]] = field(default_factory=list)
    dev_history: List[Dict[str, Any]] = field(default_factory=list)
    change_points: List[Dict[str, Any]] = field(default_factory=list)
    equipment_master: List[Dict[str, Any]] = field(default_factory=list)
    maintenance_history: List[Dict[str, Any]] = field(default_factory=list)
    fdc_parameters: List[Dict[str, Any]] = field(default_factory=list)
    recipe_data: List[Dict[str, Any]] = field(default_factory=list)
    defect_images: List[str] = field(default_factory=list)


class DefectDataExtractor:
    """
    불량 데이터 추출기

    고객 품질 불량 발생 시:
    1. 불량 셀의 제품 이력 추출
    2. 개발 단계 이력 추출
    3. 변경점 데이터 추출
    4. 관련 설비/FDC/레시피 데이터 추출
    """

    def __init__(
        self,
        greenplum_connector=None,
        oracle_connector=None,
        s3_connector=None
    ):
        self.gp = greenplum_connector
        self.oracle = oracle_connector
        self.s3 = s3_connector

    def extract_all_related_data(
        self,
        defect_case: DefectCase,
        lookback_days: int = 90
    ) -> ExtractedData:
        """
        불량 케이스와 관련된 모든 데이터 추출

        Args:
            defect_case: 불량 케이스 정보
            lookback_days: 과거 조회 기간 (일)

        Returns:
            추출된 데이터
        """
        logger.info(f"데이터 추출 시작: Case {defect_case.case_id}, Cell {defect_case.cell_id}")

        result = ExtractedData(
            case_id=defect_case.case_id,
            cell_id=defect_case.cell_id,
            extraction_time=datetime.now()
        )

        # 1. Greenplum에서 제품 이력 추출
        if self.gp:
            gp_data = self.gp.extract_defect_history(
                cell_id=defect_case.cell_id,
                include_dev_history=True,
                include_change_points=True
            )
            result.product_history = gp_data.get("product_history", [])
            result.dev_history = gp_data.get("dev_history", [])
            result.change_points = gp_data.get("change_points", [])
            logger.info(f"Greenplum 추출 완료: 제품이력 {len(result.product_history)}건")

        # 2. Oracle에서 설비/FDC 데이터 추출
        if self.oracle and result.product_history:
            # 관련 설비 ID 추출
            equipment_ids = list(set([
                h.get("equipment_id") for h in result.product_history
                if h.get("equipment_id")
            ]))

            # 설비 마스터
            result.equipment_master = self.oracle.get_equipment_master(equipment_ids)

            # 유지보수 이력
            start_date = (defect_case.defect_date - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
            end_date = defect_case.defect_date.strftime("%Y-%m-%d")

            for eq_id in equipment_ids:
                maint = self.oracle.get_maintenance_history(eq_id, start_date, end_date)
                result.maintenance_history.extend(maint)

            # FDC 파라미터
            for hist in result.product_history[:5]:  # 최근 5개 공정만
                if hist.get("equipment_id") and hist.get("process_time"):
                    fdc = self.oracle.get_fdc_parameters(
                        hist["equipment_id"],
                        hist["process_time"]
                    )
                    result.fdc_parameters.extend(fdc)

            # 레시피 데이터
            recipe_ids = list(set([
                h.get("recipe_id") for h in result.product_history
                if h.get("recipe_id")
            ]))
            result.recipe_data = self.oracle.get_recipe_master(recipe_ids)

            logger.info(f"Oracle 추출 완료: 설비 {len(equipment_ids)}개, FDC {len(result.fdc_parameters)}건")

        # 3. S3에서 불량 이미지 조회
        if self.s3:
            date_prefix = defect_case.defect_date.strftime("%Y/%m/%d")
            image_prefix = f"defect-images/{date_prefix}"
            images = self.s3.list_parquet_files(image_prefix)
            result.defect_images = [img["key"] for img in images if defect_case.cell_id in img["key"]]
            logger.info(f"S3 이미지 조회 완료: {len(result.defect_images)}개")

        logger.info(f"데이터 추출 완료: Case {defect_case.case_id}")
        return result

    def extract_similar_cases(
        self,
        defect_type: str,
        lookback_days: int = 180,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        유사 불량 케이스 추출

        Args:
            defect_type: 결함 유형
            lookback_days: 과거 조회 기간
            limit: 최대 조회 건수

        Returns:
            유사 케이스 목록
        """
        if not self.gp:
            return self._simulate_similar_cases(defect_type, limit)

        sql = f"""
        SELECT
            case_id, cell_id, defect_type, defect_date,
            customer, severity, root_cause, action_taken,
            resolution_time_hours
        FROM qms.defect_cases
        WHERE defect_type = '{defect_type}'
          AND defect_date >= CURRENT_DATE - INTERVAL '{lookback_days} days'
          AND root_cause IS NOT NULL
        ORDER BY defect_date DESC
        LIMIT {limit}
        """
        return self.gp.query(sql)

    def extract_wafer_map_data(self, lot_id: str, wafer_id: str) -> Dict[str, Any]:
        """
        웨이퍼 맵 데이터 추출

        Args:
            lot_id: 로트 ID
            wafer_id: 웨이퍼 ID

        Returns:
            웨이퍼 맵 데이터
        """
        if not self.gp:
            return self._simulate_wafer_map(lot_id, wafer_id)

        sql = f"""
        SELECT
            lot_id, wafer_id, die_x, die_y,
            bin_code, test_result, defect_code,
            inspection_time
        FROM qms.wafer_map
        WHERE lot_id = '{lot_id}' AND wafer_id = '{wafer_id}'
        ORDER BY die_x, die_y
        """
        die_data = self.gp.query(sql)

        return {
            "lot_id": lot_id,
            "wafer_id": wafer_id,
            "die_count": len(die_data),
            "die_data": die_data,
            "defect_count": len([d for d in die_data if d.get("defect_code")])
        }

    def _simulate_similar_cases(self, defect_type: str, limit: int) -> List[Dict[str, Any]]:
        """시뮬레이션 유사 케이스"""
        return [
            {
                "case_id": f"CASE2024{i:04d}",
                "cell_id": f"CELL{i:03d}",
                "defect_type": defect_type,
                "defect_date": f"2024-{12-i:02d}-{15+i:02d}",
                "customer": "Customer_A",
                "severity": "MEDIUM",
                "root_cause": "Temperature variation in coating process",
                "action_taken": "Adjusted temperature control parameters",
                "resolution_time_hours": 24 + i * 2
            }
            for i in range(min(limit, 5))
        ]

    def _simulate_wafer_map(self, lot_id: str, wafer_id: str) -> Dict[str, Any]:
        """시뮬레이션 웨이퍼 맵"""
        import random
        die_data = []
        for x in range(10):
            for y in range(10):
                die_data.append({
                    "lot_id": lot_id,
                    "wafer_id": wafer_id,
                    "die_x": x,
                    "die_y": y,
                    "bin_code": "01" if random.random() > 0.1 else "08",
                    "test_result": "PASS" if random.random() > 0.1 else "FAIL",
                    "defect_code": None if random.random() > 0.1 else "DEF001",
                    "inspection_time": "2025-01-01 10:00:00"
                })

        return {
            "lot_id": lot_id,
            "wafer_id": wafer_id,
            "die_count": len(die_data),
            "die_data": die_data,
            "defect_count": len([d for d in die_data if d.get("defect_code")])
        }
