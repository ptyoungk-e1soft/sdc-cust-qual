"""
Apache Spark 분석 프로세서
이기종 데이터 통합 및 분석용 데이터마트 구성
"""

import logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class SparkConfig:
    """Spark 설정"""
    app_name: str = "DisplayDefectAnalysis"
    master: str = "local[*]"
    executor_memory: str = "4g"
    driver_memory: str = "2g"
    enable_aqe: bool = True  # Adaptive Query Execution
    broadcast_threshold: int = 10485760  # 10MB


class SparkProcessor:
    """
    Apache Spark 분석 프로세서

    기능:
    - 이기종 데이터 통합 (S3 + Oracle)
    - 분석용 데이터마트 구성
    - AQE 및 브로드캐스트 조인 최적화
    """

    def __init__(self, config: Optional[SparkConfig] = None):
        self.config = config or SparkConfig()
        self._spark = None

    def get_spark_session(self):
        """Spark 세션 생성/반환"""
        if self._spark:
            return self._spark

        try:
            from pyspark.sql import SparkSession

            builder = SparkSession.builder \
                .appName(self.config.app_name) \
                .master(self.config.master) \
                .config("spark.executor.memory", self.config.executor_memory) \
                .config("spark.driver.memory", self.config.driver_memory)

            # AQE 활성화
            if self.config.enable_aqe:
                builder = builder \
                    .config("spark.sql.adaptive.enabled", "true") \
                    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
                    .config("spark.sql.adaptive.skewJoin.enabled", "true")

            # 브로드캐스트 조인 임계값
            builder = builder.config(
                "spark.sql.autoBroadcastJoinThreshold",
                self.config.broadcast_threshold
            )

            self._spark = builder.getOrCreate()
            logger.info(f"Spark 세션 생성: {self.config.app_name}")
            return self._spark

        except ImportError:
            logger.warning("PySpark가 설치되지 않음. 시뮬레이션 모드로 실행")
            return None

    def stop(self):
        """Spark 세션 종료"""
        if self._spark:
            self._spark.stop()
            self._spark = None
            logger.info("Spark 세션 종료")

    def build_analysis_datamart(
        self,
        case_id: str,
        parquet_files: Dict[str, Path],
        manufacturing_data: Optional[List[Dict]] = None
    ) -> Path:
        """
        분석용 데이터마트 구성

        Args:
            case_id: 케이스 ID
            parquet_files: Parquet 파일 경로 딕셔너리
            manufacturing_data: 추가 제조현장 데이터

        Returns:
            데이터마트 출력 경로
        """
        spark = self.get_spark_session()
        output_path = Path(f"/tmp/datamart/{case_id}")
        output_path.mkdir(parents=True, exist_ok=True)

        if not spark:
            # 시뮬레이션 모드
            return self._simulate_datamart(case_id, parquet_files, output_path)

        try:
            dataframes = {}

            # Parquet 파일 로드
            for table_name, file_path in parquet_files.items():
                if file_path and file_path.exists():
                    df = spark.read.parquet(str(file_path))
                    dataframes[table_name] = df
                    df.createOrReplaceTempView(table_name)
                    logger.info(f"로드: {table_name} ({df.count()}건)")

            # 제조현장 데이터 추가
            if manufacturing_data:
                mfg_df = spark.createDataFrame(manufacturing_data)
                mfg_df.createOrReplaceTempView("manufacturing_data")
                dataframes["manufacturing_data"] = mfg_df

            # 통합 분석 뷰 생성
            self._create_analysis_views(spark, dataframes)

            # 결과 저장
            analysis_df = spark.sql("""
                SELECT * FROM defect_analysis_view
            """)
            analysis_df.write.mode("overwrite").parquet(str(output_path / "analysis_result"))

            logger.info(f"데이터마트 생성 완료: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"데이터마트 생성 실패: {e}")
            return self._simulate_datamart(case_id, parquet_files, output_path)

    def _create_analysis_views(self, spark, dataframes: Dict):
        """분석 뷰 생성"""
        # 제품 이력 + 변경점 조인
        if "product_history" in dataframes and "change_points" in dataframes:
            spark.sql("""
                CREATE OR REPLACE TEMP VIEW product_with_changes AS
                SELECT
                    ph.*,
                    cp.change_type,
                    cp.change_desc,
                    cp.change_date
                FROM product_history ph
                LEFT JOIN change_points cp
                    ON ph.lot_id = cp.lot_id
                    OR ph.cell_id = cp.cell_id
            """)

        # FDC 이상 탐지
        if "fdc_parameters" in dataframes:
            spark.sql("""
                CREATE OR REPLACE TEMP VIEW fdc_anomalies AS
                SELECT
                    *,
                    CASE
                        WHEN param_value > upper_limit THEN 'HIGH'
                        WHEN param_value < lower_limit THEN 'LOW'
                        ELSE 'NORMAL'
                    END AS anomaly_status
                FROM fdc_parameters
            """)

        # 통합 분석 뷰
        spark.sql("""
            CREATE OR REPLACE TEMP VIEW defect_analysis_view AS
            SELECT
                pc.cell_id,
                pc.lot_id,
                pc.process_step,
                pc.equipment_id,
                pc.process_time,
                pc.change_type,
                pc.change_desc,
                fa.param_name,
                fa.param_value,
                fa.anomaly_status
            FROM product_with_changes pc
            LEFT JOIN fdc_anomalies fa
                ON pc.equipment_id = fa.equipment_id
        """)

    def run_correlation_analysis(
        self,
        case_id: str,
        target_variable: str = "defect_type"
    ) -> Dict[str, Any]:
        """
        상관관계 분석 실행

        Args:
            case_id: 케이스 ID
            target_variable: 타겟 변수

        Returns:
            분석 결과
        """
        spark = self.get_spark_session()
        if not spark:
            return self._simulate_correlation_analysis()

        try:
            # 상관관계 분석 쿼리
            result = spark.sql(f"""
                SELECT
                    equipment_id,
                    process_step,
                    COUNT(*) as count,
                    AVG(param_value) as avg_value,
                    STDDEV(param_value) as std_value
                FROM defect_analysis_view
                GROUP BY equipment_id, process_step
                ORDER BY count DESC
            """).collect()

            return {
                "case_id": case_id,
                "analysis_type": "correlation",
                "results": [row.asDict() for row in result]
            }
        except Exception as e:
            logger.error(f"상관관계 분석 실패: {e}")
            return self._simulate_correlation_analysis()

    def _simulate_datamart(
        self,
        case_id: str,
        parquet_files: Dict[str, Path],
        output_path: Path
    ) -> Path:
        """시뮬레이션 데이터마트"""
        import json

        summary = {
            "case_id": case_id,
            "created_at": datetime.now().isoformat(),
            "source_files": {k: str(v) for k, v in parquet_files.items() if v},
            "status": "simulated"
        }

        summary_path = output_path / "summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        logger.info(f"[시뮬레이션] 데이터마트 생성: {output_path}")
        return output_path

    def _simulate_correlation_analysis(self) -> Dict[str, Any]:
        """시뮬레이션 상관관계 분석"""
        return {
            "analysis_type": "correlation",
            "status": "simulated",
            "results": [
                {"equipment_id": "EQ001", "process_step": "TFT", "count": 150, "avg_value": 350.5, "std_value": 2.3},
                {"equipment_id": "EQ002", "process_step": "CF", "count": 120, "avg_value": 280.0, "std_value": 1.8},
                {"equipment_id": "EQ003", "process_step": "CELL", "count": 100, "avg_value": 25.0, "std_value": 0.5}
            ]
        }
