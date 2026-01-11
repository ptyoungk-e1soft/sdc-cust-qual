# SDC Customer Quality System - 문서 가이드

**버전**: 2.0
**최종 수정일**: 2026-01-04

---

## 문서 목록

본 디렉토리에는 SDC Customer Quality System의 주요 문서들이 포함되어 있습니다.

| 문서 | 파일명 | 대상 | 설명 |
|------|--------|------|------|
| **사용자 매뉴얼** | [USER_MANUAL.md](./USER_MANUAL.md) | 일반 사용자 | 시스템 사용 방법 안내 |
| **운영 매뉴얼** | [OPERATION_MANUAL.md](./OPERATION_MANUAL.md) | 시스템 관리자 | 시스템 설치, 운영, 관리 방법 |
| **기능 설명서** | [FUNCTION_SPECIFICATION.md](./FUNCTION_SPECIFICATION.md) | 개발자 | 상세 기능 및 API 명세 |
| **시나리오 가이드** | [SCENARIO_GUIDE.md](./SCENARIO_GUIDE.md) | 전체 | 업무 시나리오별 상세 사용 가이드 |

---

## 문서별 상세 내용

### 1. 사용자 매뉴얼 (USER_MANUAL.md)

일반 사용자를 위한 시스템 사용 가이드입니다.

**주요 내용:**
- 시스템 개요 및 접속 방법
- 메뉴 구조 및 화면 구성
- CS 워크플로우 사용법
  - Step 1. 불만 접수
  - Step 2. 1차 분석
  - Step 3. 2차 분석
  - Step 4. 보고서 생성
  - 결재 현황 대시보드
- 빅데이터 분석
- 결함 분석 (VLM)
- 품질 대시보드
- GraphRAG 관리
- 그래프 시각화
- 목업 데이터 생성
- FAQ

---

### 2. 운영 매뉴얼 (OPERATION_MANUAL.md)

시스템 관리자를 위한 설치 및 운영 가이드입니다.

**주요 내용:**
- 시스템 요구사항 (HW/SW)
- 설치 및 설정
  - Python 환경 설정
  - Neo4j 설정
  - GPU 설정
- 시스템 시작/중지/재시작
- 설정 관리
  - 서버 포트
  - 결재 라인
  - 결재 기한
  - 고객사/부서 정보
- 데이터 관리
- 로그 관리
- 백업 및 복구
- 모니터링
- 트러블슈팅
- 보안 관리

---

### 3. 기능 설명서 (FUNCTION_SPECIFICATION.md)

개발자를 위한 상세 기능 및 API 명세입니다.

**주요 내용:**
- 시스템 아키텍처
- 모듈별 기능 설명
  - CS 워크플로우 기능
  - 결재 시스템 기능
  - VLM 분석 기능
  - GraphRAG 기능
  - 빅데이터 분석 기능
  - 대시보드 기능
  - 이메일/문서 생성 기능
- API 레퍼런스
- 상수 정의

---

### 4. 시나리오 가이드 (SCENARIO_GUIDE.md)

실제 업무 시나리오를 기반으로 한 상세 사용 가이드입니다.

**주요 시나리오:**

| 시나리오 | 설명 |
|----------|------|
| **시나리오 1** | 고객 불만 접수 및 처리 (End-to-End) |
| **시나리오 2** | 결재 프로세스 (다단계 승인) |
| **시나리오 3** | AI 기반 결함 분석 (VLM) |
| **시나리오 4** | 지식그래프 활용 (GraphRAG) |
| **시나리오 5** | 빅데이터 품질 분석 (Spark) |
| **시나리오 6** | 품질 현황 모니터링 (Dashboard) |
| **통합 시나리오** | 전체 프로세스 End-to-End |

**주요 내용:**
- 시나리오별 배경 및 목표
- 단계별 상세 조작 방법
- 입력값 및 결과 예시
- 생성되는 산출물 목록
- 체크리스트

---

## 시스템 개요

```
┌─────────────────────────────────────────────────────────────┐
│              SDC Customer Quality System v2.0                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   [이미지 입력] → [VLM 추론] → [GraphRAG] → [분석 결과]      │
│                    Cosmos      Neo4j                         │
│                    Reason 7B   지식그래프                    │
│                                                              │
│   [CS 불만접수] → [1차분석] → [2차분석] → [보고서생성]        │
│                    품질부서    귀책부서    최종보고           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 메뉴 구조

```
SDC Customer Quality System
├── 1. CS 워크플로우
│   ├── Step 1. 불만 접수
│   ├── Step 2. 1차 분석
│   ├── Step 3. 2차 분석
│   ├── Step 4. 보고서 생성
│   └── 결재 현황
├── 2. 빅데이터 분석
├── 3. 결함 분석
├── 4. 품질 대시보드
├── 5. GraphRAG 관리
├── 6. 그래프 시각화
├── 7. 목업 데이터
└── 8. 시스템 정보
```

---

## 기술 스택

| 구성요소 | 기술 | 버전 |
|----------|------|------|
| VLM | Cosmos Reason 7B | NVIDIA |
| Fine-tuning | LoRA (PEFT) | r=64, alpha=128 |
| Graph DB | Neo4j | 5.x |
| BigData | Apache Spark | 3.x |
| API | FastAPI | - |
| UI | Gradio | 4.x |
| Container | Docker | - |
| Document | python-docx | - |

---

## 빠른 시작

### 시스템 접속

```
URL: http://localhost:7860
```

### 시스템 시작

```bash
# 가상환경 활성화
source .venv/bin/activate

# 데모 시작
python3 demo/app_full.py
```

### 시스템 중지

```bash
pkill -f "python.*app_full.py"
```

---

## 디렉토리 구조

```
/home/ptyoung/sdc-cust-qual/
├── demo/
│   └── app_full.py          # 메인 애플리케이션
├── docs/                     # 문서
│   ├── README.md             # 문서 가이드 (본 문서)
│   ├── USER_MANUAL.md        # 사용자 매뉴얼
│   ├── OPERATION_MANUAL.md   # 운영 매뉴얼
│   └── FUNCTION_SPECIFICATION.md  # 기능 설명서
├── data/                     # 데이터 파일
├── configs/                  # 설정 파일
├── output/                   # 모델 출력
└── .venv/                    # 가상환경
```

---

## 런타임 디렉토리

```
/tmp/
├── cs_emails/                    # 발송된 이메일 로그
├── cs_approval_notifications/    # 알림 발송 로그
├── cs_approval_logs/             # 결재 로그
├── cs_reports/                   # 생성된 보고서
│   ├── first_output/             # 1차 산출물 보고서
│   ├── second_output/            # 2차 산출물 보고서
│   └── final/                    # 최종 보고서
└── mockdata/                     # 목업 데이터
```

---

## 문의 및 지원

- **개발팀**: 품질관리시스템개발팀
- **이메일**: qms-support@sdc.com

---

## 변경 이력

| 버전 | 날짜 | 변경 내용 |
|------|------|----------|
| 2.0 | 2026-01-04 | 결재 시스템, 기한 알림 기능 추가 |
| 1.5 | 2025-12-20 | CS 워크플로우 기능 추가 |
| 1.0 | 2025-12-01 | 초기 버전 |

---

*본 문서는 SDC Customer Quality System v2.0 기준으로 작성되었습니다.*
