"""
디스플레이 결함 분석 시스템 - 데모 인터페이스
Cosmos Reason VLM + GraphRAG 기반
"""

import gradio as gr
import json
import re
from pathlib import Path
from PIL import Image
import sys

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

# 샘플 이미지 경로
SAMPLE_DIR = Path(__file__).parent.parent / "data" / "processed"
SAMPLE_IMAGES = list(SAMPLE_DIR.glob("*.png"))[:10] if SAMPLE_DIR.exists() else []


def parse_model_response(response: str) -> dict:
    """모델 응답을 파싱하여 구조화된 결과 반환"""
    result = {
        "thinking": "",
        "defect_type": "",
        "location": "",
        "severity": "",
        "cause": "",
        "action": "",
        "raw_response": response,
    }

    # <think> 태그 파싱
    think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    if think_match:
        result["thinking"] = think_match.group(1).strip()

    # <answer> 태그 파싱
    answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if answer_match:
        answer_text = answer_match.group(1)

        # 각 필드 추출
        patterns = {
            "defect_type": r"결함\s*유형[:\s]*([^\n]+)",
            "location": r"위치[:\s]*([^\n]+)",
            "severity": r"심각도[:\s]*([^\n]+)",
            "cause": r"(?:가능한\s*)?원인[:\s]*([^\n]+)",
            "action": r"(?:권장\s*)?조치[:\s]*([^\n]+)",
        }

        for key, pattern in patterns.items():
            match = re.search(pattern, answer_text)
            if match:
                result[key] = match.group(1).strip()

    return result


def get_severity_color(severity: str) -> str:
    """심각도에 따른 색상 반환"""
    severity_lower = severity.lower()
    if "high" in severity_lower or "높" in severity_lower:
        return "#ff4444"
    elif "medium" in severity_lower or "중" in severity_lower:
        return "#ffaa00"
    else:
        return "#44aa44"


def get_severity_emoji(severity: str) -> str:
    """심각도에 따른 이모지 반환"""
    severity_lower = severity.lower()
    if "high" in severity_lower or "높" in severity_lower:
        return "🔴"
    elif "medium" in severity_lower or "중" in severity_lower:
        return "🟡"
    else:
        return "🟢"


def create_result_html(parsed: dict) -> str:
    """분석 결과를 HTML로 포맷팅"""
    severity_color = get_severity_color(parsed["severity"])
    severity_emoji = get_severity_emoji(parsed["severity"])

    html = f"""
    <div style="font-family: 'Noto Sans KR', sans-serif; padding: 20px; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); border-radius: 15px; color: white;">

        <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px; margin-bottom: 15px;">
            <h3 style="margin: 0 0 10px 0; color: #00d4ff;">🔍 AI 추론 과정</h3>
            <p style="margin: 0; line-height: 1.6; color: #ccc; font-style: italic;">
                "{parsed['thinking']}"
            </p>
        </div>

        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px;">
            <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px;">
                <div style="color: #888; font-size: 12px; margin-bottom: 5px;">결함 유형</div>
                <div style="font-size: 18px; font-weight: bold; color: #fff;">
                    ⚠️ {parsed['defect_type']}
                </div>
            </div>

            <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px;">
                <div style="color: #888; font-size: 12px; margin-bottom: 5px;">위치</div>
                <div style="font-size: 18px; font-weight: bold; color: #fff;">
                    📍 {parsed['location']}
                </div>
            </div>

            <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px;">
                <div style="color: #888; font-size: 12px; margin-bottom: 5px;">심각도</div>
                <div style="font-size: 18px; font-weight: bold; color: {severity_color};">
                    {severity_emoji} {parsed['severity'].upper()}
                </div>
            </div>

            <div style="background: rgba(255,255,255,0.1); padding: 15px; border-radius: 10px;">
                <div style="color: #888; font-size: 12px; margin-bottom: 5px;">추정 원인</div>
                <div style="font-size: 16px; color: #fff;">
                    🔧 {parsed['cause']}
                </div>
            </div>
        </div>

        <div style="background: linear-gradient(90deg, #00d4ff22, #00d4ff11); padding: 15px; border-radius: 10px; margin-top: 15px; border-left: 4px solid #00d4ff;">
            <div style="color: #00d4ff; font-size: 14px; font-weight: bold; margin-bottom: 5px;">
                💡 권장 조치
            </div>
            <div style="font-size: 16px; color: #fff;">
                {parsed['action']}
            </div>
        </div>
    </div>
    """
    return html


def create_ontology_html(parsed: dict) -> str:
    """GraphRAG 온톨로지 시각화 HTML"""
    html = f"""
    <div style="font-family: 'Noto Sans KR', sans-serif; padding: 20px; background: #0d1117; border-radius: 15px; color: white;">
        <h3 style="color: #58a6ff; margin-bottom: 20px;">🔗 지식 그래프 연결</h3>

        <div style="display: flex; align-items: center; justify-content: center; flex-wrap: wrap; gap: 10px;">
            <div style="background: #238636; padding: 10px 20px; border-radius: 20px; font-weight: bold;">
                {parsed['defect_type']}
            </div>
            <div style="color: #58a6ff; font-size: 24px;">→</div>
            <div style="background: #1f6feb; padding: 10px 20px; border-radius: 20px;">
                CAUSED_BY
            </div>
            <div style="color: #58a6ff; font-size: 24px;">→</div>
            <div style="background: #da3633; padding: 10px 20px; border-radius: 20px; font-weight: bold;">
                {parsed['cause']}
            </div>
        </div>

        <div style="margin-top: 20px; padding: 15px; background: rgba(255,255,255,0.05); border-radius: 10px;">
            <div style="color: #8b949e; font-size: 12px; margin-bottom: 10px;">관련 노드</div>
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                <span style="background: #21262d; padding: 5px 15px; border-radius: 15px; font-size: 14px;">📍 {parsed['location']}</span>
                <span style="background: #21262d; padding: 5px 15px; border-radius: 15px; font-size: 14px;">⚙️ 공정 파라미터</span>
                <span style="background: #21262d; padding: 5px 15px; border-radius: 15px; font-size: 14px;">🔧 {parsed['action']}</span>
            </div>
        </div>
    </div>
    """
    return html


# 모델 로딩 상태
MODEL_LOADED = False
model = None
processor = None


def load_model():
    """모델 로드 (실제 GPU 환경에서만 동작)"""
    global MODEL_LOADED, model, processor

    if MODEL_LOADED:
        return True

    try:
        import torch
        from transformers import AutoProcessor
        from peft import PeftModel

        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as VLModel
        except ImportError:
            try:
                from transformers import Qwen2VLForConditionalGeneration as VLModel
            except ImportError:
                from transformers import AutoModelForVision2Seq as VLModel

        print("Loading base model...")
        base_model = VLModel.from_pretrained(
            "nvidia/Cosmos-Reason1-7B",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        print("Loading LoRA adapter...")
        model = PeftModel.from_pretrained(
            base_model,
            "output/display_defect/checkpoint-200",
            torch_dtype=torch.bfloat16,
        )
        model.eval()

        print("Loading processor...")
        processor = AutoProcessor.from_pretrained(
            "nvidia/Cosmos-Reason1-7B",
            trust_remote_code=True,
        )

        MODEL_LOADED = True
        return True

    except Exception as e:
        print(f"Model loading failed: {e}")
        return False


def analyze_image_real(image):
    """실제 모델을 사용한 이미지 분석"""
    global model, processor

    if not MODEL_LOADED:
        if not load_model():
            return "모델 로딩 실패", "", ""

    import torch

    question = "이 디스플레이 패널 검사 이미지를 분석하여 결함 유형을 분류하고 근본 원인을 추론해주세요."

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    try:
        from qwen_vl_utils import process_vision_info
        image_inputs, video_inputs = process_vision_info(messages)
    except:
        image_inputs = [image]
        video_inputs = None

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response = processor.decode(generated_ids, skip_special_tokens=True)

    parsed = parse_model_response(response)
    result_html = create_result_html(parsed)
    ontology_html = create_ontology_html(parsed)

    return result_html, ontology_html, response


def analyze_image_demo(image):
    """데모용 시뮬레이션 분석 (GPU 없이 동작)"""
    import random

    defect_types = ["라인 결함", "휘점 결함", "데드 픽셀", "무라 (불균일)", "스크래치", "이물질"]
    locations = ["좌측 상단", "중앙부", "우측 하단", "중앙 상단", "좌측 하단"]
    severities = ["high", "medium", "low"]
    causes = [
        "전극 터치 패턴과의 상호작용",
        "백라이트 불균일",
        "TFT 구동 회로 결함",
        "전하 누적",
        "치구 접촉",
        "클린룸 환경 오염",
    ]
    actions = [
        "균일한 전극 패턴 확인",
        "백라이트 조정",
        "TFT 검사 강화",
        "절연 공정 파라미터 조정",
        "보호 필름 적용",
        "클린룸 청정도 점검",
    ]

    idx = random.randint(0, len(defect_types) - 1)

    parsed = {
        "thinking": f"이미지를 분석한 결과, {random.choice(locations)} 영역에서 {defect_types[idx]} 패턴이 관찰됩니다. 결함의 형태와 분포를 고려할 때, {causes[idx]}이 원인으로 추정됩니다.",
        "defect_type": defect_types[idx],
        "location": random.choice(locations),
        "severity": random.choice(severities),
        "cause": causes[idx],
        "action": actions[idx],
    }

    raw_response = f"""<think>{parsed['thinking']}</think>
<answer>
결함 유형: {parsed['defect_type']}
위치: {parsed['location']}
심각도: {parsed['severity']}
가능한 원인: {parsed['cause']}
권장 조치: {parsed['action']}
</answer>"""

    result_html = create_result_html(parsed)
    ontology_html = create_ontology_html(parsed)

    return result_html, ontology_html, raw_response


def analyze_image(image, use_real_model: bool = False):
    """이미지 분석 메인 함수"""
    if image is None:
        return (
            "<div style='padding: 20px; text-align: center; color: #666;'>이미지를 업로드해주세요.</div>",
            "",
            "",
        )

    if use_real_model:
        return analyze_image_real(image)
    else:
        return analyze_image_demo(image)


# Gradio 인터페이스 생성
def create_demo():
    with gr.Blocks(
        title="디스플레이 결함 분석 시스템",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="cyan",
        ),
        css="""
        .gradio-container {
            max-width: 1400px !important;
        }
        .header {
            text-align: center;
            padding: 20px;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border-radius: 15px;
            margin-bottom: 20px;
        }
        .header h1 {
            color: #00d4ff;
            margin: 0;
        }
        .header p {
            color: #888;
            margin: 10px 0 0 0;
        }
        """,
    ) as demo:

        # 헤더
        gr.HTML("""
        <div class="header">
            <h1>🖥️ 디스플레이 결함 분석 시스템</h1>
            <p>Cosmos Reason VLM + GraphRAG 기반 지능형 품질 검사</p>
        </div>
        """)

        with gr.Row():
            # 왼쪽: 이미지 입력
            with gr.Column(scale=1):
                gr.Markdown("### 📤 이미지 입력")
                image_input = gr.Image(
                    type="pil",
                    label="검사 이미지",
                    height=400,
                )

                with gr.Row():
                    use_real = gr.Checkbox(
                        label="실제 모델 사용 (GPU 필요)",
                        value=False,
                    )
                    analyze_btn = gr.Button(
                        "🔍 분석 시작",
                        variant="primary",
                        size="lg",
                    )

                # 샘플 이미지 갤러리
                if SAMPLE_IMAGES:
                    gr.Markdown("#### 샘플 이미지")
                    sample_gallery = gr.Gallery(
                        value=[(str(img), img.name) for img in SAMPLE_IMAGES[:6]],
                        columns=3,
                        height=150,
                        label="클릭하여 선택",
                    )

            # 오른쪽: 분석 결과
            with gr.Column(scale=1):
                gr.Markdown("### 📊 분석 결과")
                result_output = gr.HTML(
                    label="분석 결과",
                )

                gr.Markdown("### 🔗 GraphRAG 온톨로지")
                ontology_output = gr.HTML(
                    label="온톨로지",
                )

        # Raw 응답 (접기 가능)
        with gr.Accordion("🔧 Raw 모델 응답", open=False):
            raw_output = gr.Textbox(
                label="원본 응답",
                lines=10,
            )

        # 시스템 아키텍처 설명
        with gr.Accordion("📐 시스템 아키텍처", open=False):
            gr.Markdown("""
            ## 시스템 구성

            ```
            ┌─────────────────────────────────────────────────────────────────┐
            │                    디스플레이 결함 분석 시스템                      │
            ├─────────────────────────────────────────────────────────────────┤
            │                                                                  │
            │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
            │  │   이미지     │ -> │  VLM 추론    │ -> │  결과 파싱    │       │
            │  │   입력       │    │  (7B 모델)   │    │  & 시각화    │       │
            │  └──────────────┘    └──────────────┘    └──────────────┘       │
            │                            │                                     │
            │                            ▼                                     │
            │                    ┌──────────────┐                             │
            │                    │  GraphRAG    │                             │
            │                    │  온톨로지     │                             │
            │                    │  (Neo4j)     │                             │
            │                    └──────────────┘                             │
            │                            │                                     │
            │                            ▼                                     │
            │                    ┌──────────────┐                             │
            │                    │  근본원인     │                             │
            │                    │  추론 엔진    │                             │
            │                    └──────────────┘                             │
            └─────────────────────────────────────────────────────────────────┘
            ```

            ### 핵심 기술

            | 구성요소 | 기술 | 설명 |
            |---------|------|------|
            | **VLM** | Cosmos Reason 7B | NVIDIA의 추론 특화 비전-언어 모델 |
            | **Fine-tuning** | LoRA (r=64) | 경량화된 파인튜닝으로 도메인 특화 |
            | **GraphRAG** | Neo4j | 결함-원인-공정 관계 그래프 |
            | **API** | FastAPI | RESTful API 서버 |
            | **배포** | Docker + GPU | 컨테이너 기반 배포 |
            """)

        # 이벤트 연결
        analyze_btn.click(
            fn=analyze_image,
            inputs=[image_input, use_real],
            outputs=[result_output, ontology_output, raw_output],
        )

        # 갤러리에서 이미지 선택 시
        if SAMPLE_IMAGES:
            def load_sample(evt: gr.SelectData):
                return Image.open(SAMPLE_IMAGES[evt.index])

            sample_gallery.select(
                fn=load_sample,
                outputs=image_input,
            )

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
