#!/usr/bin/env python
"""데이터 준비 스크립트"""

import argparse
import json
from pathlib import Path

from src.data.llava_converter import LLaVAConverter, generate_synthetic_dataset
from src.data.dataset import create_train_val_split
from src.data.preprocessing import ImagePreprocessor


def main():
    parser = argparse.ArgumentParser(description="디스플레이 결함 데이터 준비")

    subparsers = parser.add_subparsers(dest="command", help="명령")

    # convert 명령
    convert_parser = subparsers.add_parser("convert", help="데이터 형식 변환")
    convert_parser.add_argument(
        "--input", "-i", required=True, help="입력 파일 경로"
    )
    convert_parser.add_argument(
        "--output", "-o", required=True, help="출력 파일 경로"
    )
    convert_parser.add_argument(
        "--format", "-f",
        choices=["json", "csv", "cvat", "label_studio", "coco"],
        default="json",
        help="입력 형식",
    )
    convert_parser.add_argument(
        "--image-dir", help="이미지 디렉토리 (json 형식의 경우)"
    )

    # split 명령
    split_parser = subparsers.add_parser("split", help="train/val/test 분할")
    split_parser.add_argument(
        "--input", "-i", required=True, help="입력 파일 경로"
    )
    split_parser.add_argument(
        "--output-dir", "-o", required=True, help="출력 디렉토리"
    )
    split_parser.add_argument(
        "--val-ratio", type=float, default=0.1, help="검증 세트 비율"
    )
    split_parser.add_argument(
        "--test-ratio", type=float, default=0.1, help="테스트 세트 비율"
    )
    split_parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")

    # synthetic 명령
    synthetic_parser = subparsers.add_parser("synthetic", help="합성 데이터 생성")
    synthetic_parser.add_argument(
        "--output", "-o", required=True, help="출력 파일 경로"
    )
    synthetic_parser.add_argument(
        "--num-samples", "-n", type=int, default=100, help="샘플 수"
    )
    synthetic_parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")

    # preprocess 명령
    preprocess_parser = subparsers.add_parser("preprocess", help="이미지 전처리")
    preprocess_parser.add_argument(
        "--input-dir", "-i", required=True, help="입력 이미지 디렉토리"
    )
    preprocess_parser.add_argument(
        "--output-dir", "-o", required=True, help="출력 이미지 디렉토리"
    )
    preprocess_parser.add_argument(
        "--size", type=int, default=1024, help="출력 이미지 크기"
    )
    preprocess_parser.add_argument(
        "--pattern", default="*.png", help="이미지 파일 패턴"
    )

    args = parser.parse_args()

    if args.command == "convert":
        convert_data(args)
    elif args.command == "split":
        split_data(args)
    elif args.command == "synthetic":
        generate_synthetic(args)
    elif args.command == "preprocess":
        preprocess_images(args)
    else:
        parser.print_help()


def convert_data(args):
    """데이터 형식 변환"""
    converter = LLaVAConverter()

    print(f"변환 중: {args.input} -> {args.output}")

    if args.format == "json":
        if not args.image_dir:
            print("Error: --image-dir required for json format")
            return
        converter.convert_custom_json(args.input, args.output, args.image_dir)

    elif args.format == "csv":
        converter.convert_csv(args.input, args.output)

    elif args.format == "cvat":
        converter.convert_cvat_xml(args.input, args.output)

    elif args.format == "label_studio":
        converter.convert_label_studio(args.input, args.output)

    elif args.format == "coco":
        converter.convert_coco(args.input, args.output)

    print(f"변환 완료: {args.output}")


def split_data(args):
    """데이터 분할"""
    print(f"분할 중: {args.input}")

    train_path, val_path, test_path = create_train_val_split(
        args.input,
        args.output_dir,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    print(f"Train: {train_path}")
    print(f"Val: {val_path}")
    print(f"Test: {test_path}")


def generate_synthetic(args):
    """합성 데이터 생성"""
    print(f"합성 데이터 생성 중: {args.num_samples}개")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    generate_synthetic_dataset(
        args.output,
        num_samples=args.num_samples,
        seed=args.seed,
    )

    print(f"생성 완료: {args.output}")


def preprocess_images(args):
    """이미지 전처리"""
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    preprocessor = ImagePreprocessor(target_size=(args.size, args.size))

    image_files = list(input_dir.glob(args.pattern))
    print(f"전처리 중: {len(image_files)}개 이미지")

    for image_path in image_files:
        try:
            image = preprocessor.process(image_path)
            output_path = output_dir / image_path.name
            image.save(output_path)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    print(f"전처리 완료: {output_dir}")


if __name__ == "__main__":
    main()
