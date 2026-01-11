"""데이터 모듈 테스트"""

import json
import tempfile
from pathlib import Path

import pytest

from src.data.preprocessing import ImagePreprocessor
from src.data.augmentation import DefectAugmentor
from src.data.llava_converter import LLaVAConverter, generate_synthetic_dataset
from src.data.constants import DEFECT_TYPES


class TestImagePreprocessor:
    """이미지 전처리 테스트"""

    def test_init(self):
        preprocessor = ImagePreprocessor(target_size=(512, 512))
        assert preprocessor.target_size == (512, 512)

    def test_default_size(self):
        preprocessor = ImagePreprocessor()
        assert preprocessor.target_size == (1024, 1024)


class TestDefectAugmentor:
    """데이터 증강 테스트"""

    def test_init(self):
        augmentor = DefectAugmentor(seed=42)
        assert len(augmentor.augmentations) > 0


class TestLLaVAConverter:
    """LLaVA 변환기 테스트"""

    def test_init(self):
        converter = LLaVAConverter()
        assert converter.system_prompt is not None
        assert converter.user_prompt is not None

    def test_percent_to_location(self):
        converter = LLaVAConverter()

        # 중앙
        assert "중앙" in converter._percent_to_location(50, 50)

        # 좌측 상단
        loc = converter._percent_to_location(10, 10)
        assert "좌측" in loc or "상단" in loc


class TestSyntheticDataGeneration:
    """합성 데이터 생성 테스트"""

    def test_generate_synthetic(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            output_path = f.name

        generate_synthetic_dataset(output_path, num_samples=10, seed=42)

        with open(output_path, encoding="utf-8") as f:
            data = json.load(f)

        assert len(data) == 10
        assert "conversations" in data[0]
        assert "image" in data[0]

        Path(output_path).unlink()


class TestDefectTypes:
    """결함 유형 테스트"""

    def test_defect_types_defined(self):
        assert "dead_pixel" in DEFECT_TYPES
        assert "mura" in DEFECT_TYPES
        assert DEFECT_TYPES["dead_pixel"] == "데드 픽셀"
