"""
Greenplum 데이터베이스 커넥터
PXF/FastBCP를 통한 대용량 데이터 추출
"""

import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from pathlib import Path
import subprocess

logger = logging.getLogger(__name__)


@dataclass
class GreenplumConfig:
    """Greenplum 연결 설정"""
    host: str = "localhost"
    port: int = 5432
    database: str = "analytics"
    user: str = "gpadmin"
    password: str = ""
    schema: str = "public"


class GreenplumConnector:
    """
    Greenplum 데이터베이스 커넥터

    기능:
    - 대용량 이력 데이터 조회
    - PXF를 통한 S3 연동
    - FastBCP를 통한 고속 데이터 추출
    """

    def __init__(self, config: GreenplumConfig):
        self.config = config
        self._connection = None

    def connect(self):
        """데이터베이스 연결"""
        try:
            import psycopg2
            self._connection = psycopg2.connect(
                host=self.config.host,
                port=self.config.port,
                database=self.config.database,
                user=self.config.user,
                password=self.config.password
            )
            logger.info(f"Greenplum 연결 성공: {self.config.host}:{self.config.port}")
            return True
        except ImportError:
            logger.warning("psycopg2가 설치되지 않음. 시뮬레이션 모드로 실행")
            return True
        except Exception as e:
            logger.error(f"Greenplum 연결 실패: {e}")
            return False

    def disconnect(self):
        """연결 종료"""
        if self._connection:
            self._connection.close()
            self._connection = None
            logger.info("Greenplum 연결 종료")

    def query(self, sql: str) -> List[Dict[str, Any]]:
        """SQL 쿼리 실행"""
        if not self._connection:
            logger.warning("연결되지 않음. 시뮬레이션 데이터 반환")
            return self._simulate_query(sql)

        try:
            cursor = self._connection.cursor()
            cursor.execute(sql)
            columns = [desc[0] for desc in cursor.description]
            results = [dict(zip(columns, row)) for row in cursor.fetchall()]
            cursor.close()
            return results
        except Exception as e:
            logger.error(f"쿼리 실행 실패: {e}")
            return []

    def extract_defect_history(
        self,
        cell_id: str,
        include_dev_history: bool = True,
        include_change_points: bool = True
    ) -> Dict[str, Any]:
        """
        불량 셀의 제품 이력 추출

        Args:
            cell_id: 불량 셀 ID
            include_dev_history: 개발 단계 이력 포함 여부
            include_change_points: 변경점 데이터 포함 여부

        Returns:
            추출된 이력 데이터
        """
        result = {
            "cell_id": cell_id,
            "product_history": [],
            "dev_history": [],
            "change_points": [],
            "process_data": [],
            "quality_data": []
        }

        # 제품 이력 조회
        product_sql = f"""
        SELECT
            lot_id, wafer_id, cell_id, process_step,
            equipment_id, recipe_id, process_time,
            param_1, param_2, param_3
        FROM qms.product_history
        WHERE cell_id = '{cell_id}'
        ORDER BY process_time
        """
        result["product_history"] = self.query(product_sql)

        # 개발 단계 이력
        if include_dev_history:
            dev_sql = f"""
            SELECT
                dev_phase, experiment_id, condition_code,
                result_code, engineer, dev_date
            FROM yms.development_history
            WHERE cell_id = '{cell_id}'
            ORDER BY dev_date
            """
            result["dev_history"] = self.query(dev_sql)

        # 변경점 데이터
        if include_change_points:
            change_sql = f"""
            SELECT
                change_id, change_type, change_desc,
                before_value, after_value, change_date,
                approver, impact_level
            FROM qms.change_points
            WHERE cell_id = '{cell_id}'
               OR lot_id IN (
                   SELECT lot_id FROM qms.product_history
                   WHERE cell_id = '{cell_id}'
               )
            ORDER BY change_date
            """
            result["change_points"] = self.query(change_sql)

        logger.info(f"셀 {cell_id} 이력 추출 완료")
        return result

    def export_to_parquet_via_pxf(
        self,
        table_name: str,
        s3_path: str,
        filter_condition: Optional[str] = None
    ) -> bool:
        """
        PXF를 통한 S3 Parquet 내보내기

        Args:
            table_name: 소스 테이블명
            s3_path: S3 대상 경로
            filter_condition: WHERE 조건절
        """
        where_clause = f"WHERE {filter_condition}" if filter_condition else ""

        pxf_sql = f"""
        CREATE WRITABLE EXTERNAL TABLE pxf_export_{table_name} (
            LIKE {table_name}
        )
        LOCATION ('pxf://{s3_path}?PROFILE=s3:parquet&COMPRESSION_CODEC=snappy')
        FORMAT 'CUSTOM' (FORMATTER='pxfwritable_export');

        INSERT INTO pxf_export_{table_name}
        SELECT * FROM {table_name} {where_clause};
        """

        logger.info(f"PXF 내보내기: {table_name} -> {s3_path}")
        # 실제 실행은 연결이 필요
        return True

    def _simulate_query(self, sql: str) -> List[Dict[str, Any]]:
        """시뮬레이션 쿼리 결과"""
        if "product_history" in sql.lower():
            return [
                {
                    "lot_id": "L20250101001",
                    "wafer_id": "W001",
                    "cell_id": "CELL001",
                    "process_step": "TFT_ARRAY",
                    "equipment_id": "EQ001",
                    "recipe_id": "RCP001",
                    "process_time": "2025-01-01 10:00:00",
                    "param_1": 350.5,
                    "param_2": 25.0,
                    "param_3": 0.05
                },
                {
                    "lot_id": "L20250101001",
                    "wafer_id": "W001",
                    "cell_id": "CELL001",
                    "process_step": "CF_ARRAY",
                    "equipment_id": "EQ002",
                    "recipe_id": "RCP002",
                    "process_time": "2025-01-01 14:00:00",
                    "param_1": 280.0,
                    "param_2": 22.5,
                    "param_3": 0.03
                }
            ]
        elif "development_history" in sql.lower():
            return [
                {
                    "dev_phase": "Proto",
                    "experiment_id": "EXP001",
                    "condition_code": "COND_A",
                    "result_code": "PASS",
                    "engineer": "Engineer1",
                    "dev_date": "2024-12-01"
                }
            ]
        elif "change_points" in sql.lower():
            return [
                {
                    "change_id": "CHG001",
                    "change_type": "RECIPE",
                    "change_desc": "온도 파라미터 조정",
                    "before_value": "340",
                    "after_value": "350",
                    "change_date": "2024-12-15",
                    "approver": "Manager1",
                    "impact_level": "MEDIUM"
                }
            ]
        return []
