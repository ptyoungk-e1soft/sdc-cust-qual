"""LLaVA 형식 변환 모듈

다양한 라벨링 도구 형식을 LLaVA 대화 형식으로 변환
지원 형식: CVAT, Label Studio, COCO, 커스텀 CSV/JSON
"""

import csv
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from .constants import DEFECT_TYPES


class LLaVAConverter:
    """LLaVA 대화 형식 변환기"""

    def __init__(
        self,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        answer_template: str | None = None,
    ):
        self.system_prompt = system_prompt or (
            "당신은 디스플레이 품질 검사 전문가입니다. "
            "이미지를 분석하여 결함 유형을 분류하고 근본 원인을 추론해주세요. "
            "<think>추론 과정</think><answer>분석 결과</answer> 형식으로 응답하세요."
        )

        self.user_prompt = user_prompt or (
            "<image>\n이 디스플레이 패널 검사 이미지를 분석하여 "
            "결함 유형을 분류하고 근본 원인을 추론해주세요."
        )

        self.answer_template = answer_template or (
            "<think>{reasoning}</think>\n"
            "<answer>\n"
            "결함 유형: {defect_type}\n"
            "위치: {location}\n"
            "심각도: {severity}\n"
            "가능한 원인: {root_cause}\n"
            "권장 조치: {action}\n"
            "</answer>"
        )

    def _create_conversation(self, annotation: dict) -> list[dict]:
        """단일 어노테이션을 대화 형식으로 변환"""
        defect_type = annotation.get("defect_type", "unknown")
        korean_type = DEFECT_TYPES.get(defect_type, defect_type)

        answer = self.answer_template.format(
            reasoning=annotation.get("reasoning", "이미지를 분석하고 있습니다..."),
            defect_type=korean_type,
            location=annotation.get("location", "확인 필요"),
            severity=annotation.get("severity", "medium"),
            root_cause=annotation.get("root_cause", "추가 분석 필요"),
            action=annotation.get("action", "상세 검토 권장"),
        )

        return [
            {"from": "human", "value": self.user_prompt},
            {"from": "gpt", "value": answer},
        ]

    def convert_custom_json(
        self,
        input_path: str | Path,
        output_path: str | Path,
        image_dir: str | Path,
    ) -> None:
        """커스텀 JSON 형식 변환

        입력 형식:
        [
            {
                "image": "image_001.png",
                "defect_type": "dead_pixel",
                "location": "중앙 상단",
                "severity": "medium",
                "root_cause": "TFT 제조 결함",
                "action": "장비 점검",
                "reasoning": "..."
            }
        ]
        """
        with open(input_path, encoding="utf-8") as f:
            annotations = json.load(f)

        image_dir = Path(image_dir)
        output_data = []

        for ann in annotations:
            image_path = ann.get("image", "")

            # 이미지 존재 확인
            full_path = image_dir / image_path
            if not full_path.exists():
                continue

            sample = {
                "conversations": self._create_conversation(ann),
                "image": image_path,
            }
            output_data.append(sample)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

    def convert_csv(
        self,
        input_path: str | Path,
        output_path: str | Path,
    ) -> None:
        """CSV 형식 변환

        필수 컬럼: image, defect_type
        선택 컬럼: location, severity, root_cause, action, reasoning
        """
        output_data = []

        with open(input_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sample = {
                    "conversations": self._create_conversation(row),
                    "image": row["image"],
                }
                output_data.append(sample)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

    def convert_cvat_xml(
        self,
        input_path: str | Path,
        output_path: str | Path,
        defect_label_map: dict[str, str] | None = None,
    ) -> None:
        """CVAT XML 형식 변환"""
        tree = ET.parse(input_path)
        root = tree.getroot()

        defect_label_map = defect_label_map or {}
        output_data = []

        for image_elem in root.findall(".//image"):
            image_name = image_elem.get("name", "")

            # 바운딩 박스/폴리곤에서 라벨 추출
            for box in image_elem.findall(".//box"):
                label = box.get("label", "")
                defect_type = defect_label_map.get(label, label)

                # 위치 계산
                xtl, ytl = float(box.get("xtl", 0)), float(box.get("ytl", 0))
                xbr, ybr = float(box.get("xbr", 0)), float(box.get("ybr", 0))
                location = self._calculate_location(xtl, ytl, xbr, ybr)

                annotation = {
                    "defect_type": defect_type,
                    "location": location,
                    "severity": "medium",
                    "root_cause": "",
                    "action": "",
                    "reasoning": f"{label} 결함이 감지되었습니다.",
                }

                sample = {
                    "conversations": self._create_conversation(annotation),
                    "image": image_name,
                }
                output_data.append(sample)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

    def convert_label_studio(
        self,
        input_path: str | Path,
        output_path: str | Path,
        defect_label_map: dict[str, str] | None = None,
    ) -> None:
        """Label Studio JSON 형식 변환"""
        with open(input_path, encoding="utf-8") as f:
            data = json.load(f)

        defect_label_map = defect_label_map or {}
        output_data = []

        for task in data:
            image_path = task.get("data", {}).get("image", "")
            if image_path.startswith("/data/"):
                image_path = image_path.replace("/data/local-files/?d=", "")

            annotations = task.get("annotations", [])
            for ann in annotations:
                results = ann.get("result", [])
                for result in results:
                    if result.get("type") == "rectanglelabels":
                        labels = result.get("value", {}).get("rectanglelabels", [])
                        if labels:
                            label = labels[0]
                            defect_type = defect_label_map.get(label, label)

                            # 위치 계산 (퍼센트 → 설명)
                            value = result.get("value", {})
                            x = value.get("x", 50)
                            y = value.get("y", 50)
                            location = self._percent_to_location(x, y)

                            annotation_data = {
                                "defect_type": defect_type,
                                "location": location,
                                "severity": "medium",
                                "root_cause": "",
                                "action": "",
                                "reasoning": f"{label} 결함이 감지되었습니다.",
                            }

                            sample = {
                                "conversations": self._create_conversation(annotation_data),
                                "image": Path(image_path).name,
                            }
                            output_data.append(sample)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

    def convert_coco(
        self,
        input_path: str | Path,
        output_path: str | Path,
    ) -> None:
        """COCO 형식 변환"""
        with open(input_path, encoding="utf-8") as f:
            coco_data = json.load(f)

        # 카테고리 ID → 이름 매핑
        categories = {cat["id"]: cat["name"] for cat in coco_data.get("categories", [])}

        # 이미지 ID → 파일명 매핑
        images = {img["id"]: img["file_name"] for img in coco_data.get("images", [])}

        output_data = []

        for ann in coco_data.get("annotations", []):
            image_id = ann.get("image_id")
            category_id = ann.get("category_id")

            image_name = images.get(image_id, "")
            defect_type = categories.get(category_id, "unknown")

            # bbox: [x, y, width, height]
            bbox = ann.get("bbox", [0, 0, 0, 0])
            location = self._calculate_location(
                bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]
            )

            annotation_data = {
                "defect_type": defect_type,
                "location": location,
                "severity": "medium",
                "root_cause": "",
                "action": "",
                "reasoning": f"{defect_type} 결함이 감지되었습니다.",
            }

            sample = {
                "conversations": self._create_conversation(annotation_data),
                "image": image_name,
            }
            output_data.append(sample)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

    def _calculate_location(
        self, x1: float, y1: float, x2: float, y2: float, image_size: tuple[int, int] = (1024, 1024)
    ) -> str:
        """바운딩 박스 좌표를 위치 설명으로 변환"""
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2

        # 정규화
        norm_x = center_x / image_size[0]
        norm_y = center_y / image_size[1]

        return self._percent_to_location(norm_x * 100, norm_y * 100)

    def _percent_to_location(self, x_percent: float, y_percent: float) -> str:
        """퍼센트 좌표를 위치 설명으로 변환"""
        # 수평 위치
        if x_percent < 33:
            h_pos = "좌측"
        elif x_percent > 66:
            h_pos = "우측"
        else:
            h_pos = "중앙"

        # 수직 위치
        if y_percent < 33:
            v_pos = "상단"
        elif y_percent > 66:
            v_pos = "하단"
        else:
            v_pos = "중앙"

        if h_pos == "중앙" and v_pos == "중앙":
            return "중앙부"
        elif h_pos == "중앙":
            return v_pos
        elif v_pos == "중앙":
            return h_pos
        else:
            return f"{h_pos} {v_pos}"


def generate_synthetic_dataset(
    output_path: str | Path,
    num_samples: int = 100,
    seed: int = 42,
) -> None:
    """합성 학습 데이터셋 생성 (테스트용)"""
    import random

    random.seed(seed)

    defect_types = list(DEFECT_TYPES.keys())
    severities = ["low", "medium", "high", "critical"]
    locations = ["좌측 상단", "중앙 상단", "우측 상단", "좌측", "중앙부", "우측", "좌측 하단", "중앙 하단", "우측 하단"]

    root_causes = {
        "dead_pixel": ["TFT 제조 결함", "회로 단선", "트랜지스터 손상"],
        "bright_spot": ["전하 누적", "절연층 결함", "Gate 누설"],
        "line_defect": ["Driver IC 불량", "TAB 본딩 불량", "라인 단락"],
        "mura": ["증착 불균일", "백라이트 불균일", "액정 주입 불량"],
        "scratch": ["핸들링 손상", "운반 중 마찰", "치구 접촉"],
        "particle": ["클린룸 오염", "재료 이물질", "장비 파티클"],
        "custom": ["복합 원인", "추가 분석 필요"],
    }

    actions = {
        "dead_pixel": ["TFT 공정 장비 점검", "트랜지스터 테스트 강화"],
        "bright_spot": ["증착 장비 점검", "절연 공정 파라미터 조정"],
        "line_defect": ["Driver IC 로트 검사", "본딩 장비 점검"],
        "mura": ["증착 두께 균일도 점검", "백라이트 조정"],
        "scratch": ["핸들링 절차 교육", "보호 필름 적용"],
        "particle": ["클린룸 환경 점검", "필터 교체"],
        "custom": ["상세 분석 진행"],
    }

    converter = LLaVAConverter()
    output_data = []

    for i in range(num_samples):
        defect_type = random.choice(defect_types)
        severity = random.choice(severities)
        location = random.choice(locations)
        root_cause = random.choice(root_causes.get(defect_type, ["확인 필요"]))
        action = random.choice(actions.get(defect_type, ["추가 검토"]))

        korean_type = DEFECT_TYPES[defect_type]
        reasoning = (
            f"이미지를 분석한 결과, {location} 영역에서 {korean_type} 패턴이 관찰됩니다. "
            f"결함의 형태와 분포를 고려할 때, {root_cause}이 원인으로 추정됩니다."
        )

        annotation = {
            "defect_type": defect_type,
            "location": location,
            "severity": severity,
            "root_cause": root_cause,
            "action": action,
            "reasoning": reasoning,
        }

        sample = {
            "conversations": converter._create_conversation(annotation),
            "image": f"synthetic_{i:05d}.png",
        }
        output_data.append(sample)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
