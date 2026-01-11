#!/usr/bin/env python
"""모델 평가 스크립트"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.pipeline import create_pipeline
from src.inference.postprocess import BatchReportGenerator


def main():
    parser = argparse.ArgumentParser(description="모델 평가")

    parser.add_argument(
        "--model", "-m",
        type=str,
        required=True,
        help="평가할 모델 경로",
    )
    parser.add_argument(
        "--test-data", "-t",
        type=str,
        required=True,
        help="테스트 데이터 경로",
    )
    parser.add_argument(
        "--media-dir",
        type=str,
        required=True,
        help="이미지 디렉토리",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="evaluation_results.json",
        help="결과 출력 경로",
    )
    parser.add_argument(
        "--ontology-config",
        type=str,
        default="configs/ontology.yaml",
        help="온톨로지 설정 경로",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="4-bit 양자화 사용",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("모델 평가")
    print("=" * 60)
    print(f"모델: {args.model}")
    print(f"테스트 데이터: {args.test_data}")
    print("=" * 60)

    # 파이프라인 생성
    pipeline = create_pipeline(
        model_path=args.model,
        ontology_config=args.ontology_config,
        quantize=args.quantize,
    )

    # 테스트 데이터 로드
    with open(args.test_data, encoding="utf-8") as f:
        test_data = json.load(f)

    media_dir = Path(args.media_dir)

    # 평가
    results = []
    correct = 0
    total = 0

    for i, sample in enumerate(test_data):
        image_path = media_dir / sample["image"]
        if not image_path.exists():
            print(f"Skip: {image_path} not found")
            continue

        # 정답 추출
        ground_truth = extract_ground_truth(sample)

        # 예측
        result = pipeline.process(image_path)

        # 비교
        is_correct = result.defect_type == ground_truth.get("defect_type", "")

        results.append({
            "image": sample["image"],
            "ground_truth": ground_truth,
            "prediction": {
                "defect_type": result.defect_type,
                "korean_name": result.korean_name,
                "confidence": result.confidence,
            },
            "correct": is_correct,
        })

        if is_correct:
            correct += 1
        total += 1

        if (i + 1) % 10 == 0:
            print(f"진행: {i + 1}/{len(test_data)}")

    # 결과 저장
    accuracy = correct / total if total > 0 else 0

    evaluation_result = {
        "model": args.model,
        "test_data": args.test_data,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "results": results,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(evaluation_result, f, ensure_ascii=False, indent=2)

    print("=" * 60)
    print(f"정확도: {accuracy * 100:.2f}% ({correct}/{total})")
    print(f"결과 저장: {args.output}")
    print("=" * 60)

    # 보고서 생성
    report_generator = BatchReportGenerator()
    analysis_results = [r for r in results]  # Convert to AnalysisResult if needed

    pipeline.shutdown()


def extract_ground_truth(sample: dict) -> dict:
    """샘플에서 정답 추출"""
    ground_truth = {}

    for conv in sample.get("conversations", []):
        if conv.get("from") == "gpt":
            value = conv.get("value", "")

            # 결함 유형 추출
            if "결함 유형:" in value:
                start = value.find("결함 유형:") + len("결함 유형:")
                end = value.find("\n", start)
                if end == -1:
                    end = len(value)
                defect_type = value[start:end].strip()
                ground_truth["defect_type"] = normalize_defect_type(defect_type)

    return ground_truth


def normalize_defect_type(defect_type: str) -> str:
    """결함 유형 정규화"""
    type_mapping = {
        "데드 픽셀": "dead_pixel",
        "휘점 결함": "bright_spot",
        "라인 결함": "line_defect",
        "무라": "mura",
        "얼룩": "mura",
        "스크래치": "scratch",
        "이물질": "particle",
    }

    for key, value in type_mapping.items():
        if key in defect_type:
            return value

    return defect_type.lower().replace(" ", "_")


if __name__ == "__main__":
    main()
