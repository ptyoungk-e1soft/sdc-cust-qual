"""데이터셋 클래스"""

import json
from pathlib import Path
from typing import Iterator

from PIL import Image

try:
    from torch.utils.data import Dataset, IterableDataset
except ImportError:
    # torch가 없으면 기본 클래스 사용
    class Dataset:
        pass
    class IterableDataset:
        pass

from .augmentation import DefectAugmentor
from .preprocessing import ImagePreprocessor
from .constants import DEFECT_TYPES, SEVERITY_LEVELS


class DefectAnnotation:
    """결함 어노테이션 데이터 클래스"""

    def __init__(
        self,
        defect_type: str,
        location: str,
        severity: str,
        root_cause: str,
        action: str,
        reasoning: str,
        bbox: tuple[int, int, int, int] | None = None,
    ):
        self.defect_type = defect_type
        self.location = location
        self.severity = severity
        self.root_cause = root_cause
        self.action = action
        self.reasoning = reasoning
        self.bbox = bbox

    @property
    def korean_type(self) -> str:
        return DEFECT_TYPES.get(self.defect_type, self.defect_type)

    def to_dict(self) -> dict:
        return {
            "defect_type": self.defect_type,
            "korean_type": self.korean_type,
            "location": self.location,
            "severity": self.severity,
            "root_cause": self.root_cause,
            "action": self.action,
            "reasoning": self.reasoning,
            "bbox": self.bbox,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DefectAnnotation":
        return cls(
            defect_type=data["defect_type"],
            location=data.get("location", ""),
            severity=data.get("severity", "medium"),
            root_cause=data.get("root_cause", ""),
            action=data.get("action", ""),
            reasoning=data.get("reasoning", ""),
            bbox=data.get("bbox"),
        )


class DisplayDefectDataset(Dataset):
    """디스플레이 결함 데이터셋"""

    def __init__(
        self,
        annotation_path: str | Path,
        media_dir: str | Path,
        preprocessor: ImagePreprocessor | None = None,
        augmentor: DefectAugmentor | None = None,
        augment: bool = False,
    ):
        self.annotation_path = Path(annotation_path)
        self.media_dir = Path(media_dir)
        self.preprocessor = preprocessor or ImagePreprocessor()
        self.augmentor = augmentor
        self.augment = augment

        self.samples = self._load_annotations()

    def _load_annotations(self) -> list[dict]:
        """어노테이션 파일 로드"""
        with open(self.annotation_path, encoding="utf-8") as f:
            data = json.load(f)
        return data

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]

        # 이미지 로드
        image_path = self.media_dir / sample["image"]
        image = self.preprocessor.process(image_path)

        # 증강 적용
        if self.augment and self.augmentor:
            image = self.augmentor.augment(image)

        # 대화 형식 반환
        return {
            "image": image,
            "image_path": str(image_path),
            "conversations": sample["conversations"],
        }

    def get_statistics(self) -> dict:
        """데이터셋 통계"""
        stats = {
            "total_samples": len(self.samples),
            "defect_types": {},
        }

        for sample in self.samples:
            # 응답에서 결함 유형 추출 시도
            for conv in sample.get("conversations", []):
                if conv.get("from") == "gpt":
                    value = conv.get("value", "")
                    for defect_type in DEFECT_TYPES:
                        if defect_type in value or DEFECT_TYPES[defect_type] in value:
                            stats["defect_types"][defect_type] = (
                                stats["defect_types"].get(defect_type, 0) + 1
                            )
                            break

        return stats


class StreamingDefectDataset(IterableDataset):
    """대용량 데이터셋을 위한 스트리밍 데이터셋"""

    def __init__(
        self,
        annotation_path: str | Path,
        media_dir: str | Path,
        preprocessor: ImagePreprocessor | None = None,
        buffer_size: int = 1000,
    ):
        self.annotation_path = Path(annotation_path)
        self.media_dir = Path(media_dir)
        self.preprocessor = preprocessor or ImagePreprocessor()
        self.buffer_size = buffer_size

    def __iter__(self) -> Iterator[dict]:
        with open(self.annotation_path, encoding="utf-8") as f:
            data = json.load(f)

        for sample in data:
            image_path = self.media_dir / sample["image"]
            if not image_path.exists():
                continue

            image = self.preprocessor.process(image_path)

            yield {
                "image": image,
                "image_path": str(image_path),
                "conversations": sample["conversations"],
            }


def create_train_val_split(
    annotation_path: str | Path,
    output_dir: str | Path,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[Path, Path, Path]:
    """데이터셋을 train/val/test로 분할"""
    import random

    random.seed(seed)

    with open(annotation_path, encoding="utf-8") as f:
        data = json.load(f)

    random.shuffle(data)

    n_total = len(data)
    n_val = int(n_total * val_ratio)
    n_test = int(n_total * test_ratio)

    test_data = data[:n_test]
    val_data = data[n_test : n_test + n_val]
    train_data = data[n_test + n_val :]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.json"
    val_path = output_dir / "val.json"
    test_path = output_dir / "test.json"

    for path, split_data in [
        (train_path, train_data),
        (val_path, val_data),
        (test_path, test_data),
    ]:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)

    return train_path, val_path, test_path
