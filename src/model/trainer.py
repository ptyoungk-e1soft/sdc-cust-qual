"""SFT 학습기"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)

# Qwen2_5_VL 또는 Qwen2VL 모델 임포트 시도
try:
    from transformers import Qwen2_5_VLForConditionalGeneration as VLModel
except ImportError:
    try:
        from transformers import Qwen2VLForConditionalGeneration as VLModel
    except ImportError:
        VLModel = AutoModelForVision2Seq
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

try:
    import toml
except ImportError:
    toml = None


@dataclass
class SFTConfig:
    """SFT 학습 설정"""

    # 모델 설정
    model_name_or_path: str = "nvidia/Cosmos-Reason1-7B"
    model_max_length: int = 4096

    # 학습 설정
    output_dir: str = "output/display_defect"
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    bf16: bool = True
    gradient_checkpointing: bool = True

    # LoRA 설정
    lora_enabled: bool = True
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    # 양자화 설정
    quantization_enabled: bool = False
    load_in_4bit: bool = True

    # 데이터 설정
    annotation_path: str = "data/sft/train.json"
    val_annotation_path: str = "data/sft/val.json"
    media_dir: str = "data/processed/"

    # 저장 설정
    save_steps: int = 500
    eval_steps: int = 100
    logging_steps: int = 10

    @classmethod
    def from_toml(cls, path: str | Path) -> "SFTConfig":
        """TOML 파일에서 설정 로드"""
        if toml is None:
            raise ImportError("toml package required: pip install toml")

        with open(path, encoding="utf-8") as f:
            config_dict = toml.load(f)

        kwargs = {}

        # train 섹션
        train = config_dict.get("train", {})
        if "epoch" in train:
            kwargs["num_train_epochs"] = train["epoch"]
        if "output_dir" in train:
            kwargs["output_dir"] = train["output_dir"]
        if "optm_lr" in train:
            kwargs["learning_rate"] = train["optm_lr"]
        if "optm_weight_decay" in train:
            kwargs["weight_decay"] = train["optm_weight_decay"]
        if "warmup_ratio" in train:
            kwargs["warmup_ratio"] = train["warmup_ratio"]
        if "save_steps" in train:
            kwargs["save_steps"] = train["save_steps"]
        if "eval_steps" in train:
            kwargs["eval_steps"] = train["eval_steps"]
        if "logging_steps" in train:
            kwargs["logging_steps"] = train["logging_steps"]
        if "gradient_accumulation_steps" in train:
            kwargs["gradient_accumulation_steps"] = train["gradient_accumulation_steps"]
        if "train_batch_per_replica" in train:
            kwargs["per_device_train_batch_size"] = train["train_batch_per_replica"]

        # policy 섹션
        policy = config_dict.get("policy", {})
        if "model_name_or_path" in policy:
            kwargs["model_name_or_path"] = policy["model_name_or_path"]
        if "model_max_length" in policy:
            kwargs["model_max_length"] = policy["model_max_length"]
        if "model_gradient_checkpointing" in policy:
            kwargs["gradient_checkpointing"] = policy["model_gradient_checkpointing"]

        # lora 섹션
        lora = config_dict.get("lora", {})
        if "enabled" in lora:
            kwargs["lora_enabled"] = lora["enabled"]
        if "r" in lora:
            kwargs["lora_r"] = lora["r"]
        if "lora_alpha" in lora:
            kwargs["lora_alpha"] = lora["lora_alpha"]
        if "lora_dropout" in lora:
            kwargs["lora_dropout"] = lora["lora_dropout"]
        if "target_modules" in lora:
            kwargs["lora_target_modules"] = lora["target_modules"]

        # quantization 섹션
        quant = config_dict.get("quantization", {})
        if "enabled" in quant:
            kwargs["quantization_enabled"] = quant["enabled"]
        if "load_in_4bit" in quant:
            kwargs["load_in_4bit"] = quant["load_in_4bit"]

        # dataset 섹션
        dataset = config_dict.get("train", {}).get("train_policy", {}).get("dataset", {})
        if "annotation_path" in dataset:
            kwargs["annotation_path"] = dataset["annotation_path"]
        if "val_annotation_path" in dataset:
            kwargs["val_annotation_path"] = dataset["val_annotation_path"]
        if "media_dir" in dataset:
            kwargs["media_dir"] = dataset["media_dir"]

        return cls(**kwargs)


class VLMDataset(Dataset):
    """VLM 학습용 데이터셋"""

    def __init__(
        self,
        annotation_path: str | Path,
        media_dir: str | Path,
        processor: Any,
        max_length: int = 4096,
    ):
        self.media_dir = Path(media_dir)
        self.processor = processor
        self.max_length = max_length

        with open(annotation_path, encoding="utf-8") as f:
            self.samples = json.load(f)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = self.samples[idx]

        # 이미지 로드
        image_path = self.media_dir / sample["image"]
        image = Image.open(image_path)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # 대화를 메시지 형식으로 변환
        conversations = sample["conversations"]
        messages = []

        for conv in conversations:
            if conv["from"] == "human":
                content = conv["value"]
                # <image> 태그를 실제 이미지로 대체
                if "<image>" in content:
                    content = content.replace("<image>\n", "").replace("<image>", "")
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image},
                                {"type": "text", "text": content},
                            ],
                        }
                    )
                else:
                    messages.append({"role": "user", "content": content})
            elif conv["from"] == "gpt":
                messages.append({"role": "assistant", "content": conv["value"]})

        # 프로세싱
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        try:
            from qwen_vl_utils import process_vision_info

            image_inputs, video_inputs = process_vision_info(messages)
        except ImportError:
            image_inputs = [image]
            video_inputs = None

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )

        # 배치 차원 제거
        result = {k: v.squeeze(0) for k, v in inputs.items()}

        # labels 추가 (input_ids와 동일, causal LM 학습용)
        if "input_ids" in result:
            result["labels"] = result["input_ids"].clone()

        return result


class VLMTrainer(Trainer):
    """VLM 학습을 위한 커스텀 Trainer"""

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """손실 계산"""
        outputs = model(**inputs)
        loss = outputs.loss

        return (loss, outputs) if return_outputs else loss


class SFTTrainer:
    """SFT 학습기"""

    def __init__(self, config: SFTConfig):
        self.config = config
        self.model = None
        self.processor = None
        self.train_dataset = None
        self.eval_dataset = None

    def setup(self) -> None:
        """학습 환경 설정"""
        # 양자화 설정
        quantization_config = None
        if self.config.quantization_enabled and self.config.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        # 모델 로드
        self.model = VLModel.from_pretrained(
            self.config.model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=quantization_config,
            trust_remote_code=True,
        )

        # 프로세서 로드
        self.processor = AutoProcessor.from_pretrained(
            self.config.model_name_or_path,
            trust_remote_code=True,
        )

        # Gradient checkpointing
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # LoRA 설정
        if self.config.lora_enabled:
            if self.config.quantization_enabled:
                self.model = prepare_model_for_kbit_training(self.model)

            lora_config = LoraConfig(
                r=self.config.lora_r,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=self.config.lora_target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

        # 데이터셋 로드
        self.train_dataset = VLMDataset(
            annotation_path=self.config.annotation_path,
            media_dir=self.config.media_dir,
            processor=self.processor,
            max_length=self.config.model_max_length,
        )

        if self.config.val_annotation_path and Path(self.config.val_annotation_path).exists():
            self.eval_dataset = VLMDataset(
                annotation_path=self.config.val_annotation_path,
                media_dir=self.config.media_dir,
                processor=self.processor,
                max_length=self.config.model_max_length,
            )

    def train(self) -> None:
        """학습 실행"""
        if self.model is None:
            self.setup()

        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type=self.config.lr_scheduler_type,
            bf16=self.config.bf16,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_strategy="steps" if self.eval_dataset else "no",
            eval_steps=self.config.eval_steps if self.eval_dataset else None,
            save_total_limit=3,
            load_best_model_at_end=True if self.eval_dataset else False,
            report_to=["tensorboard"],
            remove_unused_columns=False,
            dataloader_pin_memory=False,
        )

        trainer = VLMTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            processing_class=self.processor,
        )

        trainer.train()

        # 최종 모델 저장
        final_path = Path(self.config.output_dir) / "checkpoint-final"
        trainer.save_model(str(final_path))
        self.processor.save_pretrained(str(final_path))

    def save_config(self) -> None:
        """설정 저장"""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        config_dict = {
            "model_name_or_path": self.config.model_name_or_path,
            "lora_enabled": self.config.lora_enabled,
            "lora_r": self.config.lora_r,
            "lora_alpha": self.config.lora_alpha,
            "num_train_epochs": self.config.num_train_epochs,
            "learning_rate": self.config.learning_rate,
        }

        with open(output_dir / "training_config.json", "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2)
