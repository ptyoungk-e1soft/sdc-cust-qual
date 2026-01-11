"""Cosmos Reason VLM 래퍼"""

import os
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    BitsAndBytesConfig,
)

try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    process_vision_info = None


class CosmosReasonWrapper:
    """Cosmos Reason VLM 모델 래퍼

    Cosmos-Reason1-7B 모델을 래핑하여 디스플레이 결함 분석에 사용
    """

    DEFAULT_MODEL = "nvidia/Cosmos-Reason1-7B"

    def __init__(
        self,
        model_path: str | Path | None = None,
        device_map: str = "auto",
        torch_dtype: str = "bfloat16",
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
    ):
        self.model_path = str(model_path) if model_path else self.DEFAULT_MODEL
        self.device_map = device_map
        self.torch_dtype = getattr(torch, torch_dtype, torch.bfloat16)
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit

        self.model = None
        self.processor = None

    def load(self) -> None:
        """모델 및 프로세서 로드"""
        quantization_config = None

        if self.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=self.torch_dtype,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif self.load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
            quantization_config=quantization_config,
            trust_remote_code=True,
        )

        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
        )

    def unload(self) -> None:
        """모델 언로드"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        torch.cuda.empty_cache()

    def generate(
        self,
        image: Image.Image | str | Path,
        prompt: str,
        system_prompt: str | None = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.1,
        top_p: float = 0.9,
        do_sample: bool = False,
    ) -> str:
        """이미지와 프롬프트로 응답 생성"""
        if self.model is None or self.processor is None:
            self.load()

        # 이미지 경로면 로드
        if isinstance(image, (str, Path)):
            image = Image.open(image)
            if image.mode != "RGB":
                image = image.convert("RGB")

        # 메시지 구성
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        )

        # 프로세싱
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        if process_vision_info:
            image_inputs, video_inputs = process_vision_info(messages)
        else:
            image_inputs = [image]
            video_inputs = None

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
        ).to(self.model.device)

        # 생성
        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                do_sample=do_sample,
                pad_token_id=self.processor.tokenizer.pad_token_id,
            )

        # 입력 토큰 제외하고 디코딩
        generated_ids = outputs[:, inputs.input_ids.shape[1] :]
        response = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0]

        return response

    def analyze_defect(
        self,
        image: Image.Image | str | Path,
        additional_context: str | None = None,
    ) -> dict[str, Any]:
        """디스플레이 결함 분석"""
        system_prompt = (
            "당신은 디스플레이 품질 검사 전문가입니다. "
            "이미지를 분석하여 결함 유형을 분류하고 근본 원인을 추론해주세요. "
            "<think>추론 과정</think><answer>분석 결과</answer> 형식으로 응답하세요."
        )

        prompt = "이 디스플레이 패널 검사 이미지를 분석하여 결함 유형을 분류하고 근본 원인을 추론해주세요."

        if additional_context:
            prompt += f"\n\n추가 정보: {additional_context}"

        response = self.generate(
            image=image,
            prompt=prompt,
            system_prompt=system_prompt,
        )

        return self._parse_response(response)

    def _parse_response(self, response: str) -> dict[str, Any]:
        """응답 파싱"""
        result = {
            "raw_response": response,
            "reasoning": "",
            "answer": "",
            "defect_type": "",
            "location": "",
            "severity": "",
            "root_cause": "",
            "action": "",
        }

        # <think></think> 추출
        if "<think>" in response and "</think>" in response:
            start = response.find("<think>") + len("<think>")
            end = response.find("</think>")
            result["reasoning"] = response[start:end].strip()

        # <answer></answer> 추출
        if "<answer>" in response and "</answer>" in response:
            start = response.find("<answer>") + len("<answer>")
            end = response.find("</answer>")
            answer_text = response[start:end].strip()
            result["answer"] = answer_text

            # 구조화된 필드 추출
            for line in answer_text.split("\n"):
                line = line.strip()
                if ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip().lower()
                    value = value.strip()

                    if "결함" in key or "유형" in key:
                        result["defect_type"] = value
                    elif "위치" in key:
                        result["location"] = value
                    elif "심각" in key:
                        result["severity"] = value
                    elif "원인" in key:
                        result["root_cause"] = value
                    elif "조치" in key or "권장" in key:
                        result["action"] = value

        return result

    @property
    def device(self) -> torch.device:
        """모델 디바이스"""
        if self.model is not None:
            return next(self.model.parameters()).device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_model(
    model_path: str | None = None,
    quantize: bool = False,
) -> CosmosReasonWrapper:
    """모델 생성 헬퍼 함수"""
    wrapper = CosmosReasonWrapper(
        model_path=model_path,
        load_in_4bit=quantize,
    )
    wrapper.load()
    return wrapper
