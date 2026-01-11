"""
데이터 파이프라인
전체 데이터 처리 흐름 관리
"""

import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class PipelineStatus(Enum):
    """파이프라인 상태"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PipelineStep:
    """파이프라인 단계"""
    name: str
    status: PipelineStatus = PipelineStatus.PENDING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Any = None
    error: Optional[str] = None


@dataclass
class PipelineResult:
    """파이프라인 실행 결과"""
    case_id: str
    status: PipelineStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    steps: List[PipelineStep] = field(default_factory=list)
    parquet_files: Dict[str, Path] = field(default_factory=dict)
    datamart_path: Optional[Path] = None
    error: Optional[str] = None


class DataPipeline:
    """
    데이터 파이프라인

    처리 흐름:
    1. 데이터 추출 (Greenplum, Oracle)
    2. Parquet 변환
    3. S3 업로드
    4. Spark 데이터마트 구성
    """

    def __init__(
        self,
        extractor=None,
        converter=None,
        s3_connector=None,
        spark_processor=None
    ):
        self.extractor = extractor
        self.converter = converter
        self.s3 = s3_connector
        self.spark = spark_processor

    def run(self, defect_case) -> PipelineResult:
        """
        파이프라인 실행

        Args:
            defect_case: DefectCase 객체

        Returns:
            파이프라인 실행 결과
        """
        result = PipelineResult(
            case_id=defect_case.case_id,
            status=PipelineStatus.RUNNING,
            start_time=datetime.now()
        )

        try:
            # Step 1: 데이터 추출
            step1 = self._run_step("데이터 추출", self._extract_data, defect_case)
            result.steps.append(step1)
            if step1.status == PipelineStatus.FAILED:
                raise Exception(step1.error)

            extracted_data = step1.result

            # Step 2: Parquet 변환
            step2 = self._run_step("Parquet 변환", self._convert_to_parquet, extracted_data)
            result.steps.append(step2)
            if step2.status == PipelineStatus.FAILED:
                raise Exception(step2.error)

            result.parquet_files = step2.result

            # Step 3: S3 업로드
            step3 = self._run_step("S3 업로드", self._upload_to_s3, result.parquet_files, defect_case.case_id)
            result.steps.append(step3)

            # Step 4: 데이터마트 구성
            step4 = self._run_step("데이터마트 구성", self._build_datamart, defect_case.case_id, result.parquet_files)
            result.steps.append(step4)
            if step4.status == PipelineStatus.COMPLETED:
                result.datamart_path = step4.result

            result.status = PipelineStatus.COMPLETED
            result.end_time = datetime.now()

            logger.info(f"파이프라인 완료: {defect_case.case_id}")

        except Exception as e:
            result.status = PipelineStatus.FAILED
            result.error = str(e)
            result.end_time = datetime.now()
            logger.error(f"파이프라인 실패: {e}")

        return result

    def _run_step(self, name: str, func, *args, **kwargs) -> PipelineStep:
        """단일 단계 실행"""
        step = PipelineStep(name=name, status=PipelineStatus.RUNNING, start_time=datetime.now())

        try:
            step.result = func(*args, **kwargs)
            step.status = PipelineStatus.COMPLETED
        except Exception as e:
            step.status = PipelineStatus.FAILED
            step.error = str(e)
            logger.error(f"Step '{name}' 실패: {e}")

        step.end_time = datetime.now()
        return step

    def _extract_data(self, defect_case):
        """데이터 추출 단계"""
        if self.extractor:
            return self.extractor.extract_all_related_data(defect_case)
        else:
            # 시뮬레이션
            from ..datalake.extractor import ExtractedData
            return ExtractedData(
                case_id=defect_case.case_id,
                cell_id=defect_case.cell_id,
                extraction_time=datetime.now(),
                product_history=[
                    {"lot_id": "L001", "wafer_id": "W001", "cell_id": defect_case.cell_id,
                     "process_step": "TFT", "equipment_id": "EQ001", "process_time": "2025-01-01 10:00:00"}
                ],
                dev_history=[
                    {"dev_phase": "Proto", "experiment_id": "EXP001", "result_code": "PASS"}
                ],
                change_points=[
                    {"change_id": "CHG001", "change_type": "RECIPE", "change_desc": "온도 조정"}
                ]
            )

    def _convert_to_parquet(self, extracted_data):
        """Parquet 변환 단계"""
        if self.converter:
            return self.converter.convert_extracted_data(extracted_data)
        else:
            # 시뮬레이션
            from .parquet_converter import ParquetConverter
            converter = ParquetConverter()
            return converter.convert_extracted_data(extracted_data)

    def _upload_to_s3(self, parquet_files: Dict[str, Path], case_id: str):
        """S3 업로드 단계"""
        if not self.s3:
            logger.info("[시뮬레이션] S3 업로드 스킵")
            return {"status": "simulated"}

        uploaded = {}
        for table_name, file_path in parquet_files.items():
            if file_path and file_path.exists():
                s3_key = f"analysis/{case_id}/{file_path.name}"
                self.s3.upload_parquet(file_path, s3_key)
                uploaded[table_name] = s3_key

        return uploaded

    def _build_datamart(self, case_id: str, parquet_files: Dict[str, Path]):
        """데이터마트 구성 단계"""
        if self.spark:
            return self.spark.build_analysis_datamart(case_id, parquet_files)
        else:
            # 시뮬레이션
            output_path = Path(f"/tmp/datamart/{case_id}")
            output_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"[시뮬레이션] 데이터마트 생성: {output_path}")
            return output_path

    def get_pipeline_summary(self, result: PipelineResult) -> Dict[str, Any]:
        """파이프라인 결과 요약"""
        duration = None
        if result.end_time and result.start_time:
            duration = (result.end_time - result.start_time).total_seconds()

        return {
            "case_id": result.case_id,
            "status": result.status.value,
            "duration_seconds": duration,
            "steps": [
                {
                    "name": step.name,
                    "status": step.status.value,
                    "duration": (step.end_time - step.start_time).total_seconds()
                    if step.end_time and step.start_time else None,
                    "error": step.error
                }
                for step in result.steps
            ],
            "parquet_files": {k: str(v) for k, v in result.parquet_files.items() if v},
            "datamart_path": str(result.datamart_path) if result.datamart_path else None,
            "error": result.error
        }
