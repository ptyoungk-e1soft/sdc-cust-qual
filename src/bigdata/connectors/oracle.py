"""
Oracle 데이터베이스 커넥터
실시간 트랜잭션 데이터 연동
"""

import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


@dataclass
class OracleConfig:
    """Oracle 연결 설정"""
    host: str = "localhost"
    port: int = 1521
    service_name: str = "ORCL"
    user: str = "system"
    password: str = ""


class OracleConnector:
    """
    Oracle 데이터베이스 커넥터

    기능:
    - 실시간 트랜잭션 데이터 조회
    - ERP 유지보수 이력 연동
    - 마스터 데이터 조회
    """

    def __init__(self, config: OracleConfig):
        self.config = config
        self._connection = None

    def connect(self):
        """데이터베이스 연결"""
        try:
            import cx_Oracle
            dsn = cx_Oracle.makedsn(
                self.config.host,
                self.config.port,
                service_name=self.config.service_name
            )
            self._connection = cx_Oracle.connect(
                user=self.config.user,
                password=self.config.password,
                dsn=dsn
            )
            logger.info(f"Oracle 연결 성공: {self.config.host}:{self.config.port}")
            return True
        except ImportError:
            logger.warning("cx_Oracle이 설치되지 않음. 시뮬레이션 모드로 실행")
            return True
        except Exception as e:
            logger.error(f"Oracle 연결 실패: {e}")
            return False

    def disconnect(self):
        """연결 종료"""
        if self._connection:
            self._connection.close()
            self._connection = None
            logger.info("Oracle 연결 종료")

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

    def get_equipment_master(self, equipment_ids: List[str]) -> List[Dict[str, Any]]:
        """설비 마스터 데이터 조회"""
        ids_str = ",".join([f"'{eid}'" for eid in equipment_ids])
        sql = f"""
        SELECT
            equipment_id, equipment_name, equipment_type,
            manufacturer, model, install_date,
            location, status, last_pm_date
        FROM erp.equipment_master
        WHERE equipment_id IN ({ids_str})
        """
        return self.query(sql)

    def get_maintenance_history(
        self,
        equipment_id: str,
        start_date: str,
        end_date: str
    ) -> List[Dict[str, Any]]:
        """설비 유지보수 이력 조회"""
        sql = f"""
        SELECT
            pm_id, equipment_id, pm_type, pm_desc,
            pm_date, technician, downtime_hours,
            parts_replaced, cost
        FROM erp.maintenance_history
        WHERE equipment_id = '{equipment_id}'
          AND pm_date BETWEEN TO_DATE('{start_date}', 'YYYY-MM-DD')
                          AND TO_DATE('{end_date}', 'YYYY-MM-DD')
        ORDER BY pm_date
        """
        return self.query(sql)

    def get_fdc_parameters(
        self,
        equipment_id: str,
        process_time: str,
        window_minutes: int = 30
    ) -> List[Dict[str, Any]]:
        """FDC(Fault Detection & Classification) 파라미터 조회"""
        sql = f"""
        SELECT
            equipment_id, param_name, param_value,
            param_unit, upper_limit, lower_limit,
            collect_time, status
        FROM fdc.parameters
        WHERE equipment_id = '{equipment_id}'
          AND collect_time BETWEEN
              TO_TIMESTAMP('{process_time}', 'YYYY-MM-DD HH24:MI:SS') - INTERVAL '{window_minutes}' MINUTE
              AND TO_TIMESTAMP('{process_time}', 'YYYY-MM-DD HH24:MI:SS') + INTERVAL '{window_minutes}' MINUTE
        ORDER BY collect_time
        """
        return self.query(sql)

    def get_recipe_master(self, recipe_ids: List[str]) -> List[Dict[str, Any]]:
        """레시피 마스터 데이터 조회"""
        ids_str = ",".join([f"'{rid}'" for rid in recipe_ids])
        sql = f"""
        SELECT
            recipe_id, recipe_name, version,
            process_type, target_product,
            create_date, modify_date, status
        FROM mes.recipe_master
        WHERE recipe_id IN ({ids_str})
        """
        return self.query(sql)

    def _simulate_query(self, sql: str) -> List[Dict[str, Any]]:
        """시뮬레이션 쿼리 결과"""
        if "equipment_master" in sql.lower():
            return [
                {
                    "equipment_id": "EQ001",
                    "equipment_name": "TFT Coater #1",
                    "equipment_type": "COATER",
                    "manufacturer": "TEL",
                    "model": "ACT-12",
                    "install_date": "2020-01-15",
                    "location": "FAB1-A1",
                    "status": "RUNNING",
                    "last_pm_date": "2024-12-20"
                }
            ]
        elif "maintenance_history" in sql.lower():
            return [
                {
                    "pm_id": "PM20241220001",
                    "equipment_id": "EQ001",
                    "pm_type": "PREVENTIVE",
                    "pm_desc": "정기 점검 및 부품 교체",
                    "pm_date": "2024-12-20",
                    "technician": "Tech001",
                    "downtime_hours": 4.5,
                    "parts_replaced": "PUMP, FILTER",
                    "cost": 15000
                }
            ]
        elif "fdc" in sql.lower() or "parameters" in sql.lower():
            return [
                {
                    "equipment_id": "EQ001",
                    "param_name": "TEMPERATURE",
                    "param_value": 350.5,
                    "param_unit": "C",
                    "upper_limit": 360.0,
                    "lower_limit": 340.0,
                    "collect_time": "2025-01-01 10:00:00",
                    "status": "NORMAL"
                },
                {
                    "equipment_id": "EQ001",
                    "param_name": "PRESSURE",
                    "param_value": 0.05,
                    "param_unit": "Pa",
                    "upper_limit": 0.08,
                    "lower_limit": 0.02,
                    "collect_time": "2025-01-01 10:00:00",
                    "status": "NORMAL"
                }
            ]
        elif "recipe_master" in sql.lower():
            return [
                {
                    "recipe_id": "RCP001",
                    "recipe_name": "TFT_COAT_STD",
                    "version": "3.2",
                    "process_type": "COATING",
                    "target_product": "65INCH_OLED",
                    "create_date": "2024-01-01",
                    "modify_date": "2024-12-01",
                    "status": "ACTIVE"
                }
            ]
        return []
