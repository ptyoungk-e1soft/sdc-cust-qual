#!/usr/bin/env python
"""SFT 학습 스크립트"""

import argparse
import os
import sys
from pathlib import Path

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.trainer import SFTConfig, SFTTrainer


def main():
    parser = argparse.ArgumentParser(description="Cosmos Reason VLM SFT 학습")

    parser.add_argument(
        "--config", "-c",
        type=str,
        default="configs/sft.toml",
        help="학습 설정 파일 경로",
    )

    # 설정 오버라이드
    parser.add_argument("--model", type=str, help="모델 경로")
    parser.add_argument("--output-dir", type=str, help="출력 디렉토리")
    parser.add_argument("--epochs", type=int, help="에폭 수")
    parser.add_argument("--lr", type=float, help="학습률")
    parser.add_argument("--batch-size", type=int, help="배치 크기")
    parser.add_argument("--lora-r", type=int, help="LoRA rank")
    parser.add_argument("--no-lora", action="store_true", help="LoRA 비활성화")
    parser.add_argument("--quantize", action="store_true", help="4-bit 양자화")

    # 데이터 경로
    parser.add_argument("--train-data", type=str, help="학습 데이터 경로")
    parser.add_argument("--val-data", type=str, help="검증 데이터 경로")
    parser.add_argument("--media-dir", type=str, help="이미지 디렉토리")

    args = parser.parse_args()

    # 설정 로드
    config_path = Path(args.config)
    if config_path.exists():
        print(f"설정 로드: {config_path}")
        config = SFTConfig.from_toml(config_path)
    else:
        print("기본 설정 사용")
        config = SFTConfig()

    # 인자로 오버라이드
    if args.model:
        config.model_name_or_path = args.model
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.epochs:
        config.num_train_epochs = args.epochs
    if args.lr:
        config.learning_rate = args.lr
    if args.batch_size:
        config.per_device_train_batch_size = args.batch_size
    if args.lora_r:
        config.lora_r = args.lora_r
    if args.no_lora:
        config.lora_enabled = False
    if args.quantize:
        config.quantization_enabled = True
        config.load_in_4bit = True
    if args.train_data:
        config.annotation_path = args.train_data
    if args.val_data:
        config.val_annotation_path = args.val_data
    if args.media_dir:
        config.media_dir = args.media_dir

    # 학습
    print("=" * 60)
    print("Cosmos Reason VLM SFT 학습")
    print("=" * 60)
    print(f"모델: {config.model_name_or_path}")
    print(f"출력: {config.output_dir}")
    print(f"에폭: {config.num_train_epochs}")
    print(f"학습률: {config.learning_rate}")
    print(f"LoRA: {config.lora_enabled} (r={config.lora_r})")
    print(f"양자화: {config.quantization_enabled}")
    print("=" * 60)

    trainer = SFTTrainer(config)
    trainer.save_config()
    trainer.train()

    print("학습 완료!")


if __name__ == "__main__":
    main()
