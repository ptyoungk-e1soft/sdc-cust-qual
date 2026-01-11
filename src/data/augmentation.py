"""데이터 증강 모듈"""

import random
from typing import Callable

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


class DefectAugmentor:
    """결함 이미지 데이터 증강"""

    def __init__(self, seed: int | None = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.augmentations: list[Callable[[Image.Image], Image.Image]] = [
            self.random_rotation,
            self.random_flip,
            self.random_brightness,
            self.random_contrast,
            self.random_blur,
            self.add_noise,
        ]

    def random_rotation(self, image: Image.Image, max_angle: float = 15.0) -> Image.Image:
        """랜덤 회전"""
        angle = random.uniform(-max_angle, max_angle)
        return image.rotate(angle, resample=Image.Resampling.BILINEAR, expand=False, fillcolor=0)

    def random_flip(self, image: Image.Image) -> Image.Image:
        """랜덤 플립"""
        if random.random() > 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if random.random() > 0.5:
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        return image

    def random_brightness(
        self, image: Image.Image, factor_range: tuple[float, float] = (0.8, 1.2)
    ) -> Image.Image:
        """랜덤 밝기 조정"""
        factor = random.uniform(*factor_range)
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)

    def random_contrast(
        self, image: Image.Image, factor_range: tuple[float, float] = (0.8, 1.2)
    ) -> Image.Image:
        """랜덤 대비 조정"""
        factor = random.uniform(*factor_range)
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)

    def random_blur(self, image: Image.Image, radius_range: tuple[float, float] = (0, 1.5)) -> Image.Image:
        """랜덤 블러"""
        radius = random.uniform(*radius_range)
        if radius > 0.1:
            return image.filter(ImageFilter.GaussianBlur(radius=radius))
        return image

    def add_noise(self, image: Image.Image, noise_level: float = 0.02) -> Image.Image:
        """가우시안 노이즈 추가"""
        img_array = np.array(image).astype(np.float32)
        noise = np.random.normal(0, noise_level * 255, img_array.shape)
        noisy_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_array)

    def augment(
        self,
        image: Image.Image,
        num_augmentations: int = 2,
    ) -> Image.Image:
        """랜덤하게 선택된 증강 적용"""
        selected = random.sample(self.augmentations, min(num_augmentations, len(self.augmentations)))
        for aug_fn in selected:
            image = aug_fn(image)
        return image

    def generate_variants(
        self,
        image: Image.Image,
        num_variants: int = 5,
    ) -> list[Image.Image]:
        """다양한 증강 버전 생성"""
        variants = [image]  # 원본 포함
        for _ in range(num_variants - 1):
            augmented = self.augment(image.copy())
            variants.append(augmented)
        return variants


class MixupAugmentor:
    """Mixup 증강 (같은 클래스 내 이미지 혼합)"""

    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha

    def mixup(
        self,
        image1: Image.Image,
        image2: Image.Image,
        label1: dict,
        label2: dict,
    ) -> tuple[Image.Image, dict]:
        """두 이미지를 혼합"""
        lam = np.random.beta(self.alpha, self.alpha)

        arr1 = np.array(image1).astype(np.float32)
        arr2 = np.array(image2).astype(np.float32)

        # 크기가 다르면 리사이즈
        if arr1.shape != arr2.shape:
            image2 = image2.resize(image1.size, Image.Resampling.BILINEAR)
            arr2 = np.array(image2).astype(np.float32)

        mixed = lam * arr1 + (1 - lam) * arr2
        mixed_image = Image.fromarray(mixed.astype(np.uint8))

        # 라벨도 혼합 (동일 클래스면 그대로)
        mixed_label = label1.copy()
        if label1 != label2:
            mixed_label["mixup_ratio"] = lam

        return mixed_image, mixed_label


class CutoutAugmentor:
    """Cutout 증강 (일부 영역 마스킹)"""

    def __init__(
        self,
        num_holes: int = 1,
        hole_size_ratio: float = 0.1,
    ):
        self.num_holes = num_holes
        self.hole_size_ratio = hole_size_ratio

    def cutout(self, image: Image.Image) -> Image.Image:
        """랜덤 영역 마스킹"""
        img_array = np.array(image)
        h, w = img_array.shape[:2]

        hole_h = int(h * self.hole_size_ratio)
        hole_w = int(w * self.hole_size_ratio)

        for _ in range(self.num_holes):
            y = random.randint(0, h - hole_h)
            x = random.randint(0, w - hole_w)
            img_array[y : y + hole_h, x : x + hole_w] = 0

        return Image.fromarray(img_array)
