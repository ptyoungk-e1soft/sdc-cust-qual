# 디스플레이 결함 분석 시스템 - 아키텍처 문서

## 1. 시스템 개요

Cosmos Reason VLM과 GraphRAG를 결합한 지능형 디스플레이 패널 품질 검사 시스템입니다.

### 핵심 목표
- **자동 결함 탐지**: VLM 기반 이미지 분석으로 결함 자동 식별
- **근본 원인 추론**: GraphRAG 온톨로지를 통한 원인 추적
- **실시간 품질 관리**: FastAPI 기반 실시간 검사 서비스

---

## 2. 시스템 아키텍처

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        디스플레이 결함 분석 시스템                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                          Presentation Layer                         │   │
│   │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │   │
│   │  │   Gradio     │  │   FastAPI    │  │   REST API   │              │   │
│   │  │   Demo UI    │  │   Server     │  │   Client     │              │   │
│   │  └──────────────┘  └──────────────┘  └──────────────┘              │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│                                    ▼                                         │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                          Inference Layer                            │   │
│   │  ┌──────────────────────────────────────────────────────────────┐  │   │
│   │  │                    Cosmos Reason VLM                          │  │   │
│   │  │  ┌────────────┐  ┌────────────┐  ┌────────────┐              │  │   │
│   │  │  │   Vision   │  │  Language  │  │  Reasoning │              │  │   │
│   │  │  │  Encoder   │──│  Decoder   │──│   Engine   │              │  │   │
│   │  │  │ (ViT-22B)  │  │  (Qwen2)   │  │ (<think>)  │              │  │   │
│   │  │  └────────────┘  └────────────┘  └────────────┘              │  │   │
│   │  │                         │                                     │  │   │
│   │  │                    LoRA Adapter                               │  │   │
│   │  │                    (r=64, α=128)                              │  │   │
│   │  └──────────────────────────────────────────────────────────────┘  │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│                                    ▼                                         │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                          Knowledge Layer                            │   │
│   │  ┌──────────────────────────────────────────────────────────────┐  │   │
│   │  │                    GraphRAG Ontology                          │  │   │
│   │  │                                                               │  │   │
│   │  │   [Defect]──CAUSED_BY──>[RootCause]                          │  │   │
│   │  │      │                       │                                │  │   │
│   │  │      │                       │                                │  │   │
│   │  │   OCCURS_IN              OCCURS_IN                           │  │   │
│   │  │      │                       │                                │  │   │
│   │  │      ▼                       ▼                                │  │   │
│   │  │   [Process]────USES────>[Equipment]                          │  │   │
│   │  │                                                               │  │   │
│   │  └──────────────────────────────────────────────────────────────┘  │   │
│   │                         Neo4j Graph DB                             │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 핵심 구성요소

### 3.1 VLM 모델 (Cosmos Reason 7B)

| 항목 | 상세 |
|------|------|
| 베이스 모델 | `nvidia/Cosmos-Reason1-7B` |
| 파라미터 수 | 8.4B (전체) |
| Vision Encoder | ViT 기반 |
| Language Model | Qwen2 아키텍처 |
| 특징 | `<think>` 태그로 추론 과정 명시화 |

### 3.2 LoRA Fine-tuning

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| r (rank) | 64 | Low-rank 차원 |
| alpha | 128 | 스케일링 팩터 |
| dropout | 0.05 | 정규화 |
| target_modules | q, k, v, o, gate, up, down | Attention + FFN |
| 학습 파라미터 | 190M (2.24%) | 전체 대비 비율 |

### 3.3 GraphRAG 온톨로지

```yaml
노드 유형:
  - Defect: 결함 유형 (데드픽셀, 라인결함, 무라 등)
  - RootCause: 근본 원인 (전극결함, 백라이트, 오염 등)
  - Process: 제조 공정 (TFT, Cell, Module)
  - Equipment: 설비 정보
  - Material: 원자재 정보

관계 유형:
  - CAUSED_BY: 결함 → 원인
  - OCCURS_IN: 결함/원인 → 공정
  - USES: 공정 → 설비/자재
  - RELATED_TO: 연관 관계
```

---

## 4. 데이터 흐름

```
[검사 이미지]
     │
     ▼
┌─────────────────┐
│  전처리         │  • 리사이징 (최적 해상도)
│  (Preprocessing)│  • 정규화
└─────────────────┘
     │
     ▼
┌─────────────────┐
│  VLM 추론       │  • 이미지 인코딩
│  (Inference)    │  • 텍스트 생성 (<think>...<answer>)
└─────────────────┘
     │
     ▼
┌─────────────────┐
│  응답 파싱      │  • 결함유형, 위치, 심각도 추출
│  (Parsing)      │  • 구조화된 JSON 변환
└─────────────────┘
     │
     ▼
┌─────────────────┐
│  GraphRAG 조회  │  • 근본원인 탐색
│  (Query)        │  • 관련 공정/설비 연결
└─────────────────┘
     │
     ▼
[분석 결과 반환]
```

---

## 5. 프로젝트 구조

```
sdc-cust-qual/
├── configs/                    # 설정 파일
│   ├── sft.toml               # SFT 학습 설정
│   └── ontology.yaml          # 온톨로지 스키마
│
├── data/                       # 데이터
│   ├── raw/                   # 원본 이미지
│   ├── processed/             # 전처리 이미지
│   ├── annotations/           # 라벨링 데이터
│   └── sft/                   # 학습 데이터 (LLaVA 형식)
│       ├── train.json
│       ├── val.json
│       └── test.json
│
├── src/                        # 소스 코드
│   ├── data/                  # 데이터 처리
│   │   ├── preprocessing.py
│   │   └── augmentation.py
│   │
│   ├── ontology/              # GraphRAG
│   │   ├── schema.py
│   │   ├── graph_store.py
│   │   ├── knowledge_base.py
│   │   └── reasoning.py
│   │
│   ├── model/                 # 모델
│   │   ├── trainer.py         # SFT 학습기
│   │   ├── cosmos_wrapper.py
│   │   └── quantization.py
│   │
│   ├── inference/             # 추론
│   │   └── pipeline.py
│   │
│   └── api/                   # API 서버
│       ├── main.py
│       ├── routes.py
│       └── schemas.py
│
├── demo/                       # 데모 UI
│   ├── app.py                 # Gradio 앱
│   └── run_demo.sh
│
├── output/                     # 학습 결과
│   └── display_defect/
│       └── checkpoint-200/    # LoRA 어댑터
│
├── docker-compose.yml
└── Dockerfile
```

---

## 6. 기술 스택

| 카테고리 | 기술 | 버전 |
|----------|------|------|
| **AI/ML** | PyTorch | 2.6.0+ |
| | Transformers | 4.49.0 |
| | PEFT (LoRA) | 0.15.0 |
| **VLM** | Cosmos Reason | 7B |
| **Graph DB** | Neo4j | 5.x |
| **API** | FastAPI | 0.110+ |
| **UI** | Gradio | 4.x |
| **Container** | Docker | 24.x |
| **GPU** | CUDA | 12.x |

---

## 7. 배포 구성

### Docker Compose

```yaml
services:
  api:
    image: display-defect-analyzer
    ports:
      - "8000:8000"
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
    environment:
      - MODEL_PATH=/app/models/checkpoint-200

  neo4j:
    image: neo4j:5
    ports:
      - "7474:7474"
      - "7687:7687"

  demo:
    image: display-defect-analyzer
    ports:
      - "7860:7860"
    command: python demo/app.py
```

---

## 8. 성능 지표

### 학습 결과 (10 Epochs)

| 지표 | 값 |
|------|-----|
| 초기 손실 | 13.17 |
| 최종 손실 | 5.21 |
| 학습 시간 | ~77분 |
| GPU 메모리 | ~16GB |

### 추론 성능

| 지표 | 값 |
|------|-----|
| 단일 이미지 처리 | ~3초 |
| 배치 처리 (8장) | ~15초 |
| GPU 메모리 | ~14GB |

---

## 9. 향후 확장

1. **실제 데이터 학습**: 실제 불량 이미지로 추가 학습
2. **다중 결함 탐지**: 단일 이미지 내 다중 결함 동시 검출
3. **실시간 스트리밍**: 라인 카메라 실시간 연동
4. **MLOps 파이프라인**: 모델 버전 관리 및 자동 배포
