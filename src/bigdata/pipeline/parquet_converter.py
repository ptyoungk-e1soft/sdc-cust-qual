"""
Parquet 변환기
데이터를 Parquet 포맷으로 변환 및 최적화
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import json

logger = logging.getLogger(__name__)


@dataclass
class ParquetConfig:
    """Parquet 설정"""
    compression: str = "snappy"  # snappy, gzip, lz4, zstd
    row_group_size: int = 100000
    output_dir: Path = Path("/tmp/parquet_output")


class ParquetConverter:
    """
    Parquet 변환기

    기능:
    - 컬럼 지향 포맷으로 변환
    - Snappy 압축 적용
    - 메타데이터 관리
    """

    def __init__(self, config: Optional[ParquetConfig] = None):
        self.config = config or ParquetConfig()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def convert_to_parquet(
        self,
        data: List[Dict[str, Any]],
        table_name: str,
        partition_cols: Optional[List[str]] = None
    ) -> Path:
        """
        데이터를 Parquet 파일로 변환

        Args:
            data: 변환할 데이터 (리스트 of 딕셔너리)
            table_name: 테이블명
            partition_cols: 파티션 컬럼 목록

        Returns:
            생성된 Parquet 파일 경로
        """
        if not data:
            logger.warning("변환할 데이터가 없습니다.")
            return None

        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            # DataFrame 생성
            table = pa.Table.from_pylist(data)

            # 출력 경로
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.config.output_dir / f"{table_name}_{timestamp}.parquet"

            # Parquet 작성
            pq.write_table(
                table,
                output_path,
                compression=self.config.compression,
                row_group_size=self.config.row_group_size
            )

            logger.info(f"Parquet 변환 완료: {output_path} ({len(data)}건)")
            return output_path

        except ImportError:
            logger.warning("pyarrow가 설치되지 않음. JSON으로 대체 저장")
            return self._save_as_json(data, table_name)

    def convert_extracted_data(self, extracted_data) -> Dict[str, Path]:
        """
        추출된 데이터 전체를 Parquet로 변환

        Args:
            extracted_data: ExtractedData 객체

        Returns:
            테이블별 Parquet 파일 경로
        """
        result = {}
        case_id = extracted_data.case_id

        # 제품 이력
        if extracted_data.product_history:
            result["product_history"] = self.convert_to_parquet(
                extracted_data.product_history,
                f"{case_id}_product_history"
            )

        # 개발 이력
        if extracted_data.dev_history:
            result["dev_history"] = self.convert_to_parquet(
                extracted_data.dev_history,
                f"{case_id}_dev_history"
            )

        # 변경점
        if extracted_data.change_points:
            result["change_points"] = self.convert_to_parquet(
                extracted_data.change_points,
                f"{case_id}_change_points"
            )

        # 설비 마스터
        if extracted_data.equipment_master:
            result["equipment_master"] = self.convert_to_parquet(
                extracted_data.equipment_master,
                f"{case_id}_equipment_master"
            )

        # 유지보수 이력
        if extracted_data.maintenance_history:
            result["maintenance_history"] = self.convert_to_parquet(
                extracted_data.maintenance_history,
                f"{case_id}_maintenance_history"
            )

        # FDC 파라미터
        if extracted_data.fdc_parameters:
            result["fdc_parameters"] = self.convert_to_parquet(
                extracted_data.fdc_parameters,
                f"{case_id}_fdc_parameters"
            )

        # 레시피 데이터
        if extracted_data.recipe_data:
            result["recipe_data"] = self.convert_to_parquet(
                extracted_data.recipe_data,
                f"{case_id}_recipe_data"
            )

        logger.info(f"케이스 {case_id} Parquet 변환 완료: {len(result)}개 파일")
        return result

    def read_parquet(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Parquet 파일 읽기

        Args:
            file_path: Parquet 파일 경로

        Returns:
            데이터 리스트
        """
        try:
            import pyarrow.parquet as pq
            table = pq.read_table(file_path)
            return table.to_pylist()
        except ImportError:
            # JSON 대체 파일 읽기
            json_path = file_path.with_suffix('.json')
            if json_path.exists():
                with open(json_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return []

    def get_parquet_schema(self, file_path: Path) -> Dict[str, str]:
        """Parquet 파일 스키마 조회"""
        try:
            import pyarrow.parquet as pq
            schema = pq.read_schema(file_path)
            return {field.name: str(field.type) for field in schema}
        except ImportError:
            return {}

    def get_parquet_metadata(self, file_path: Path) -> Dict[str, Any]:
        """Parquet 파일 메타데이터 조회"""
        try:
            import pyarrow.parquet as pq
            metadata = pq.read_metadata(file_path)
            return {
                "num_rows": metadata.num_rows,
                "num_columns": metadata.num_columns,
                "num_row_groups": metadata.num_row_groups,
                "format_version": metadata.format_version,
                "serialized_size": metadata.serialized_size
            }
        except ImportError:
            return {"status": "pyarrow not installed"}

    def _save_as_json(self, data: List[Dict[str, Any]], table_name: str) -> Path:
        """JSON으로 대체 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.config.output_dir / f"{table_name}_{timestamp}.json"

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"JSON 저장 완료: {output_path}")
        return output_path
