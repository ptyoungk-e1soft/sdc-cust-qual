"""
AWS S3 데이터 레이크 커넥터
오브젝트 스토리지 연동 및 Parquet 파일 관리
"""

import logging
import json
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, BinaryIO
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class S3Config:
    """S3 연결 설정"""
    bucket: str = "sdc-datalake"
    region: str = "ap-northeast-2"
    access_key: Optional[str] = None
    secret_key: Optional[str] = None
    endpoint_url: Optional[str] = None  # MinIO 등 호환 스토리지용


class S3Connector:
    """
    AWS S3 데이터 레이크 커넥터

    기능:
    - Parquet 파일 업로드/다운로드
    - 불량 이미지 저장
    - 로그 파일 관리
    """

    def __init__(self, config: S3Config):
        self.config = config
        self._client = None
        self._resource = None

    def connect(self):
        """S3 연결"""
        try:
            import boto3
            session_kwargs = {"region_name": self.config.region}
            if self.config.access_key and self.config.secret_key:
                session_kwargs["aws_access_key_id"] = self.config.access_key
                session_kwargs["aws_secret_access_key"] = self.config.secret_key

            session = boto3.Session(**session_kwargs)

            client_kwargs = {}
            if self.config.endpoint_url:
                client_kwargs["endpoint_url"] = self.config.endpoint_url

            self._client = session.client("s3", **client_kwargs)
            self._resource = session.resource("s3", **client_kwargs)

            logger.info(f"S3 연결 성공: {self.config.bucket}")
            return True
        except ImportError:
            logger.warning("boto3가 설치되지 않음. 시뮬레이션 모드로 실행")
            return True
        except Exception as e:
            logger.error(f"S3 연결 실패: {e}")
            return False

    def disconnect(self):
        """연결 종료"""
        self._client = None
        self._resource = None
        logger.info("S3 연결 종료")

    def upload_parquet(
        self,
        local_path: Path,
        s3_key: str,
        metadata: Optional[Dict[str, str]] = None
    ) -> bool:
        """
        Parquet 파일 업로드

        Args:
            local_path: 로컬 파일 경로
            s3_key: S3 객체 키
            metadata: 메타데이터

        Returns:
            업로드 성공 여부
        """
        if not self._client:
            logger.info(f"[시뮬레이션] Parquet 업로드: {local_path} -> s3://{self.config.bucket}/{s3_key}")
            return True

        try:
            extra_args = {"ContentType": "application/octet-stream"}
            if metadata:
                extra_args["Metadata"] = metadata

            self._client.upload_file(
                str(local_path),
                self.config.bucket,
                s3_key,
                ExtraArgs=extra_args
            )
            logger.info(f"Parquet 업로드 완료: s3://{self.config.bucket}/{s3_key}")
            return True
        except Exception as e:
            logger.error(f"Parquet 업로드 실패: {e}")
            return False

    def download_parquet(self, s3_key: str, local_path: Path) -> bool:
        """
        Parquet 파일 다운로드

        Args:
            s3_key: S3 객체 키
            local_path: 로컬 저장 경로

        Returns:
            다운로드 성공 여부
        """
        if not self._client:
            logger.info(f"[시뮬레이션] Parquet 다운로드: s3://{self.config.bucket}/{s3_key} -> {local_path}")
            return True

        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            self._client.download_file(
                self.config.bucket,
                s3_key,
                str(local_path)
            )
            logger.info(f"Parquet 다운로드 완료: {local_path}")
            return True
        except Exception as e:
            logger.error(f"Parquet 다운로드 실패: {e}")
            return False

    def upload_defect_image(
        self,
        image_data: BinaryIO,
        cell_id: str,
        defect_type: str,
        timestamp: Optional[datetime] = None
    ) -> str:
        """
        불량 이미지 업로드

        Args:
            image_data: 이미지 바이너리 데이터
            cell_id: 셀 ID
            defect_type: 결함 유형
            timestamp: 타임스탬프

        Returns:
            S3 객체 키
        """
        ts = timestamp or datetime.now()
        date_prefix = ts.strftime("%Y/%m/%d")
        filename = f"{cell_id}_{defect_type}_{ts.strftime('%H%M%S')}.png"
        s3_key = f"defect-images/{date_prefix}/{filename}"

        if not self._client:
            logger.info(f"[시뮬레이션] 이미지 업로드: s3://{self.config.bucket}/{s3_key}")
            return s3_key

        try:
            self._client.upload_fileobj(
                image_data,
                self.config.bucket,
                s3_key,
                ExtraArgs={
                    "ContentType": "image/png",
                    "Metadata": {
                        "cell_id": cell_id,
                        "defect_type": defect_type,
                        "timestamp": ts.isoformat()
                    }
                }
            )
            logger.info(f"이미지 업로드 완료: s3://{self.config.bucket}/{s3_key}")
            return s3_key
        except Exception as e:
            logger.error(f"이미지 업로드 실패: {e}")
            return ""

    def list_parquet_files(self, prefix: str) -> List[Dict[str, Any]]:
        """
        Parquet 파일 목록 조회

        Args:
            prefix: S3 경로 접두사

        Returns:
            파일 목록
        """
        if not self._client:
            # 시뮬레이션 데이터
            return [
                {
                    "key": f"{prefix}/product_history_20250101.parquet",
                    "size": 1024000,
                    "last_modified": "2025-01-01T10:00:00"
                },
                {
                    "key": f"{prefix}/change_points_20250101.parquet",
                    "size": 512000,
                    "last_modified": "2025-01-01T10:30:00"
                }
            ]

        try:
            response = self._client.list_objects_v2(
                Bucket=self.config.bucket,
                Prefix=prefix
            )

            files = []
            for obj in response.get("Contents", []):
                if obj["Key"].endswith(".parquet"):
                    files.append({
                        "key": obj["Key"],
                        "size": obj["Size"],
                        "last_modified": obj["LastModified"].isoformat()
                    })
            return files
        except Exception as e:
            logger.error(f"파일 목록 조회 실패: {e}")
            return []

    def get_parquet_metadata(self, s3_key: str) -> Dict[str, Any]:
        """Parquet 파일 메타데이터 조회"""
        if not self._client:
            return {
                "key": s3_key,
                "size": 1024000,
                "content_type": "application/octet-stream",
                "metadata": {"source": "greenplum", "table": "product_history"}
            }

        try:
            response = self._client.head_object(
                Bucket=self.config.bucket,
                Key=s3_key
            )
            return {
                "key": s3_key,
                "size": response["ContentLength"],
                "content_type": response["ContentType"],
                "metadata": response.get("Metadata", {})
            }
        except Exception as e:
            logger.error(f"메타데이터 조회 실패: {e}")
            return {}
