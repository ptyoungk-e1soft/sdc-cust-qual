"""
디스플레이 빅데이터 분석 시스템 (AI-ADAC)
===========================================

데이터 레이크 기반 고성능 분석 파이프라인

구성 요소:
- datalake: 데이터 레이크 연동 (QMS, YMS, S3)
- connectors: 데이터 소스 커넥터 (Greenplum, Oracle, S3)
- pipeline: Parquet 변환 파이프라인
- spark: Apache Spark 분석 엔진
- datamart: 분석용 데이터마트 구성
- mockdata: 시연용 목업 데이터 생성
"""

__version__ = "1.0.0"
__author__ = "SDC Customer Quality"

# Mock Data Generator export
from .mockdata import MockDataGenerator
