"""이미지 전처리 모듈"""

from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image


class ImagePreprocessor:
    """디스플레이 검사 이미지 전처리"""

    def __init__(
        self,
        target_size: Tuple[int, int] = (1024, 1024),
        normalize: bool = True,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
    ):
        self.target_size = target_size
        self.normalize = normalize
        self.mean = np.array(mean)
        self.std = np.array(std)

    def load_image(self, image_path: str | Path) -> Image.Image:
        """이미지 로드 및 RGB 변환"""
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image

    def resize(self, image: Image.Image) -> Image.Image:
        """이미지 리사이즈 (aspect ratio 유지)"""
        image.thumbnail(self.target_size, Image.Resampling.LANCZOS)

        # 패딩 추가하여 정사각형으로 만들기
        width, height = image.size
        if width != height:
            new_image = Image.new("RGB", self.target_size, (0, 0, 0))
            offset = ((self.target_size[0] - width) // 2, (self.target_size[1] - height) // 2)
            new_image.paste(image, offset)
            return new_image
        return image.resize(self.target_size, Image.Resampling.LANCZOS)

    def to_numpy(self, image: Image.Image) -> np.ndarray:
        """PIL Image를 numpy 배열로 변환"""
        arr = np.array(image).astype(np.float32) / 255.0
        if self.normalize:
            arr = (arr - self.mean) / self.std
        return arr

    def process(self, image_path: str | Path) -> Image.Image:
        """전체 전처리 파이프라인"""
        image = self.load_image(image_path)
        image = self.resize(image)
        return image

    def process_batch(self, image_paths: list[str | Path]) -> list[Image.Image]:
        """배치 전처리"""
        return [self.process(p) for p in image_paths]


class DefectRegionExtractor:
    """결함 영역 추출"""

    def __init__(self, margin: int = 50):
        self.margin = margin

    def extract_region(
        self,
        image: Image.Image,
        bbox: Tuple[int, int, int, int],
    ) -> Image.Image:
        """바운딩 박스 기준 결함 영역 추출"""
        x1, y1, x2, y2 = bbox
        width, height = image.size

        # 마진 추가
        x1 = max(0, x1 - self.margin)
        y1 = max(0, y1 - self.margin)
        x2 = min(width, x2 + self.margin)
        y2 = min(height, y2 + self.margin)

        return image.crop((x1, y1, x2, y2))

    def extract_multiple_regions(
        self,
        image: Image.Image,
        bboxes: list[Tuple[int, int, int, int]],
    ) -> list[Image.Image]:
        """복수 결함 영역 추출"""
        return [self.extract_region(image, bbox) for bbox in bboxes]


def enhance_defect_visibility(image: Image.Image, enhancement_factor: float = 1.5) -> Image.Image:
    """결함 가시성 향상 (대비 조정)"""
    from PIL import ImageEnhance

    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(enhancement_factor)


def apply_histogram_equalization(image: Image.Image) -> Image.Image:
    """히스토그램 균등화"""
    import cv2

    img_array = np.array(image)
    if len(img_array.shape) == 3:
        # LAB 색공간에서 L 채널에만 적용
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = cv2.equalizeHist(lab[:, :, 0])
        img_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    else:
        img_array = cv2.equalizeHist(img_array)

    return Image.fromarray(img_array)
