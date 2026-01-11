"""모델 양자화 모듈"""

import os
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    BitsAndBytesConfig,
)


def quantize_to_4bit(
    model_path: str | Path,
    output_path: str | Path,
    compute_dtype: str = "bfloat16",
) -> None:
    """4-bit 양자화"""
    compute_dtype_torch = getattr(torch, compute_dtype, torch.bfloat16)

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=compute_dtype_torch,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_path),
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )

    processor = AutoProcessor.from_pretrained(
        str(model_path),
        trust_remote_code=True,
    )

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(output_path))
    processor.save_pretrained(str(output_path))


def quantize_to_8bit(
    model_path: str | Path,
    output_path: str | Path,
) -> None:
    """8-bit 양자화"""
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(model_path),
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )

    processor = AutoProcessor.from_pretrained(
        str(model_path),
        trust_remote_code=True,
    )

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(output_path))
    processor.save_pretrained(str(output_path))


class FP8Quantizer:
    """FP8 양자화 (TensorRT-LLM 기반)

    실제 FP8 양자화는 TensorRT-LLM 또는 vLLM 환경에서 수행됨
    이 클래스는 FP8 변환을 위한 준비 및 메타데이터 생성을 담당
    """

    def __init__(self, model_path: str | Path):
        self.model_path = Path(model_path)

    def prepare_for_fp8(self, output_path: str | Path) -> dict[str, Any]:
        """FP8 변환 준비

        Returns:
            calibration을 위한 설정 정보
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        config = {
            "source_model": str(self.model_path),
            "target_path": str(output_path),
            "quantization": {
                "type": "fp8",
                "calibration_samples": 512,
                "calibration_method": "max",
            },
            "conversion_command": self._generate_conversion_command(output_path),
        }

        import json

        with open(output_path / "fp8_config.json", "w") as f:
            json.dump(config, f, indent=2)

        return config

    def _generate_conversion_command(self, output_path: Path) -> str:
        """TensorRT-LLM 변환 명령어 생성"""
        return f"""
# TensorRT-LLM FP8 변환 명령어
# 먼저 TensorRT-LLM 환경을 설정해야 합니다

python convert_checkpoint.py \\
    --model_dir {self.model_path} \\
    --output_dir {output_path}/trt_ckpt \\
    --dtype float16 \\
    --use_fp8_qdq \\
    --calib_size 512

trtllm-build \\
    --checkpoint_dir {output_path}/trt_ckpt \\
    --output_dir {output_path}/trt_engines \\
    --gemm_plugin float16 \\
    --max_batch_size 8 \\
    --max_input_len 2048 \\
    --max_output_len 1024
"""


def merge_lora_weights(
    base_model_path: str | Path,
    lora_path: str | Path,
    output_path: str | Path,
) -> None:
    """LoRA 가중치를 베이스 모델에 병합"""
    from peft import PeftModel

    # 베이스 모델 로드
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        str(base_model_path),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    # LoRA 가중치 로드 및 병합
    model = PeftModel.from_pretrained(model, str(lora_path))
    model = model.merge_and_unload()

    # 병합된 모델 저장
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(output_path))

    # 프로세서도 복사
    processor = AutoProcessor.from_pretrained(
        str(base_model_path),
        trust_remote_code=True,
    )
    processor.save_pretrained(str(output_path))


def estimate_memory_requirements(
    model_path: str | Path,
    batch_size: int = 1,
    sequence_length: int = 4096,
) -> dict[str, float]:
    """메모리 요구량 추정 (GB)"""
    # 7B 모델 기준 대략적인 추정
    base_params = 7e9  # 7B parameters

    estimates = {
        "fp32": base_params * 4 / 1e9,  # 4 bytes per param
        "fp16": base_params * 2 / 1e9,  # 2 bytes per param
        "bf16": base_params * 2 / 1e9,
        "int8": base_params * 1 / 1e9,  # 1 byte per param
        "int4": base_params * 0.5 / 1e9,  # 0.5 bytes per param
    }

    # KV 캐시 오버헤드 (대략적)
    kv_cache_per_token = 0.001  # GB per token per layer (추정)
    num_layers = 32  # 7B 모델 기준
    kv_overhead = sequence_length * kv_cache_per_token * num_layers * batch_size

    for key in estimates:
        estimates[key] += kv_overhead

    return estimates
