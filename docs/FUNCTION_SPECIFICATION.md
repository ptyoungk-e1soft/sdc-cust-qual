# SDC Customer Quality System - 기능 설명서

**버전**: 2.0
**최종 수정일**: 2026-01-04
**작성자**: 품질관리시스템개발팀

---

## 목차

1. [시스템 아키텍처](#1-시스템-아키텍처)
2. [CS 워크플로우 기능](#2-cs-워크플로우-기능)
3. [결재 시스템 기능](#3-결재-시스템-기능)
4. [VLM 분석 기능](#4-vlm-분석-기능)
5. [GraphRAG 기능](#5-graphrag-기능)
6. [빅데이터 분석 기능](#6-빅데이터-분석-기능)
7. [대시보드 기능](#7-대시보드-기능)
8. [이메일/문서 생성 기능](#8-이메일문서-생성-기능)
9. [API 레퍼런스](#9-api-레퍼런스)

---

## 1. 시스템 아키텍처

### 1.1 전체 아키텍처

```
┌─────────────────────────────────────────────────────────────────┐
│                        Presentation Layer                        │
│                     (Gradio Web Interface)                       │
├─────────────────────────────────────────────────────────────────┤
│                        Application Layer                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │    CS    │  │   VLM    │  │ GraphRAG │  │    BigData       │ │
│  │ Workflow │  │ Analysis │  │  Engine  │  │    Pipeline      │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                          Data Layer                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │ In-Memory│  │  Neo4j   │  │  File    │  │    Spark         │ │
│  │ Storage  │  │  Graph   │  │  System  │  │    DataLake      │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 기술 스택

| 레이어 | 기술 | 용도 |
|--------|------|------|
| Presentation | Gradio 4.x | 웹 UI |
| Application | Python 3.10+ | 비즈니스 로직 |
| AI/ML | Cosmos Reason 7B | VLM 추론 |
| AI/ML | LoRA (PEFT) | 모델 파인튜닝 |
| Graph | Neo4j 5.x | 지식그래프 |
| BigData | Apache Spark | 데이터 분석 |
| Document | python-docx | Word 문서 생성 |
| Visualization | Matplotlib, NetworkX | 차트/그래프 |

### 1.3 모듈 구성

```
app_full.py
├── Knowledge Base Module (지식베이스)
├── VLM Analysis Module (VLM 분석)
├── GraphRAG Module (지식그래프)
├── CS Workflow Module (CS 워크플로우)
├── Approval Module (결재 시스템)
├── Email Module (이메일 생성/발송)
├── Document Module (문서 생성)
├── BigData Module (빅데이터 분석)
├── Dashboard Module (대시보드)
└── Visualization Module (시각화)
```

---

## 2. CS 워크플로우 기능

### 2.1 워크플로우 초기화

#### 함수: `init_cs_workflow()`

**기능**: CS 워크플로우 시스템을 초기화합니다.

**입력**: 없음

**출력**: 초기화 결과 메시지

**처리 로직**:
1. 저장소 초기화 (complaints, analyses, reports)
2. 결재 저장소 초기화
3. 디렉토리 생성 (/tmp/cs_emails, /tmp/cs_reports)
4. 초기화 완료 메시지 반환

```python
def init_cs_workflow():
    global cs_workflow_storage
    cs_workflow_storage = {
        "complaints": {},
        "first_analyses": {},
        "second_analyses": {},
        "reports": {},
        "outputs": {},
    }
    # 디렉토리 생성
    Path("/tmp/cs_emails").mkdir(parents=True, exist_ok=True)
    Path("/tmp/cs_reports").mkdir(parents=True, exist_ok=True)
    return "CS 워크플로우 시스템이 초기화되었습니다."
```

---

### 2.2 불만 접수 생성

#### 함수: `create_cs_complaint()`

**기능**: 새로운 고객 불만 접수를 생성합니다.

**입력 파라미터**:

| 파라미터 | 타입 | 설명 |
|----------|------|------|
| customer | str | 고객사 코드 |
| product_model | str | 제품 모델 |
| lot_id | str | LOT ID |
| cell_id | str | CELL ID |
| defect_type | str | 결함 유형 |
| defect_description | str | 결함 설명 |
| severity | str | 심각도 |

**출력**: (결과 메시지, 불만접수 ID)

**처리 로직**:
1. 고유 불만접수 ID 생성 (CS-YYYYMMDD-XXX)
2. 불만접수 데이터 생성
3. 저장소에 저장
4. 결과 반환

**데이터 구조**:
```python
complaint = {
    "id": complaint_id,
    "customer": customer,
    "product_model": product_model,
    "lot_id": lot_id,
    "cell_id": cell_id,
    "defect_type": defect_type,
    "defect_description": defect_description,
    "severity": severity,
    "status": "REGISTERED",
    "created_at": datetime.now().isoformat(),
    "updated_at": datetime.now().isoformat(),
}
```

---

### 2.3 1차 분석 실행

#### 함수: `perform_first_analysis()`

**기능**: 품질부서에서 수행하는 1차 기본 분석을 실행합니다.

**입력 파라미터**:

| 파라미터 | 타입 | 설명 |
|----------|------|------|
| complaint_id | str | 불만접수 ID |

**출력**: (분석 결과 마크다운, 분석 ID)

**처리 로직**:
1. 불만접수 데이터 조회
2. 결함 유형별 분석 수행
3. 귀책 부서 결정
4. 분석 결과 저장
5. 결과 반환

**분석 내용**:
- 결함 유형 분류
- 발생 원인 추정
- 귀책 부서 결정
- 긴급도 평가
- 초기 조치사항 권고

**귀책 부서 매핑**:
```python
DEFECT_TO_DEPT = {
    "DEAD_PIXEL": "TFT공정",
    "BRIGHT_SPOT": "TFT공정",
    "LINE_DEFECT": "OLED공정",
    "MURA": "CF공정",
    "SCRATCH": "Module공정",
    "TOUCH_FAIL": "Cell공정",
}
```

---

### 2.4 2차 분석 실행

#### 함수: `perform_second_analysis()`

**기능**: 귀책부서에서 수행하는 2차 상세 분석을 실행합니다.

**입력 파라미터**:

| 파라미터 | 타입 | 설명 |
|----------|------|------|
| complaint_id | str | 불만접수 ID |
| first_analysis_id | str | 1차 분석 ID |

**출력**: (분석 결과 마크다운, 분석 ID)

**처리 로직**:
1. 불만접수 및 1차 분석 데이터 조회
2. 상세 원인 분석 수행
3. 재발 방지 대책 수립
4. 공정 개선 방안 도출
5. 분석 결과 저장
6. 결과 반환

**분석 내용**:
- 근본 원인 분석 (Root Cause Analysis)
- 재발 방지 대책
- 공정 개선 방안
- 품질 관리 강화 방안
- 검증 계획

---

### 2.5 최종 보고서 생성

#### 함수: `generate_final_report()`

**기능**: Word 형식의 최종 보고서를 생성합니다.

**입력 파라미터**:

| 파라미터 | 타입 | 설명 |
|----------|------|------|
| complaint_id | str | 불만접수 ID |

**출력**: (결과 메시지, 보고서 경로)

**처리 로직**:
1. 불만접수, 1차/2차 분석 데이터 조회
2. Word 문서 생성
3. 보고서 내용 작성
4. 파일 저장
5. 경로 반환

**보고서 구성**:
1. 표지
2. 불만 접수 정보
3. 1차 분석 결과
4. 2차 분석 결과
5. 재발 방지 대책
6. 결론 및 향후 계획

**저장 경로**: `/tmp/cs_reports/final/{complaint_id}_final_report.docx`

---

## 3. 결재 시스템 기능

### 3.1 결재 요청 생성

#### 함수: `create_approval_request()`

**기능**: 새로운 결재 요청을 생성합니다.

**입력 파라미터**:

| 파라미터 | 타입 | 설명 |
|----------|------|------|
| approval_type | str | 결재 유형 |
| complaint_id | str | 연관 불만접수 ID |
| title | str | 결재 제목 |
| content | str | 결재 내용 |
| requester | str | 요청자 (기본: "담당자") |
| custom_deadline_hours | int | 사용자 정의 기한 (선택) |

**출력**: (결재번호, 결재 데이터)

**결재 유형별 기본 기한**:
```python
APPROVAL_DEADLINES = {
    "COMPLAINT_EMAIL": 4,      # 4시간
    "MEETING_REQUEST": 8,      # 8시간
    "FIRST_ANALYSIS": 24,      # 24시간
    "TASK_ASSIGNMENT": 8,      # 8시간
    "FINAL_REPORT": 48,        # 48시간
    "CUSTOMER_REPLY": 24,      # 24시간
}
```

**결재 라인**:
```python
APPROVAL_LINE = {
    "COMPLAINT_EMAIL": [
        {"position": "팀장", "name": "이정호", "email": "jhlee@sdc.com"}
    ],
    "FINAL_REPORT": [
        {"position": "팀장", "name": "이정호", "email": "jhlee@sdc.com"},
        {"position": "부장", "name": "김대영", "email": "dykim@sdc.com"},
        {"position": "상무", "name": "박철수", "email": "cspark@sdc.com"},
    ],
    # ...
}
```

**데이터 구조**:
```python
approval = {
    "id": approval_id,
    "type": approval_type,
    "complaint_id": complaint_id,
    "title": title,
    "content": content,
    "status": "PENDING",
    "requester": requester,
    "created_at": datetime.now().isoformat(),
    "deadline": deadline.isoformat(),
    "approval_line": approval_line,
    "current_step": 0,
    "current_approver": approval_line[0],
    "history": [],
}
```

---

### 3.2 결재 처리

#### 함수: `process_approval()`

**기능**: 결재를 승인 또는 반려 처리합니다.

**입력 파라미터**:

| 파라미터 | 타입 | 설명 |
|----------|------|------|
| approval_id | str | 결재번호 |
| action | str | 처리 유형 ("APPROVE" 또는 "REJECT") |
| comment | str | 결재 의견 |
| approver_name | str | 결재자 이름 (선택) |

**출력**: (성공 여부, 결과 메시지)

**처리 로직**:
1. 결재 데이터 조회
2. 상태 확인 (PENDING 여부)
3. 승인/반려 처리
   - 승인: 다음 결재자로 이동 또는 최종 승인
   - 반려: 상태를 REJECTED로 변경
4. 이력 기록
5. 결과 반환

**상태 전이**:
```
PENDING → APPROVED (모든 결재자 승인 시)
PENDING → REJECTED (반려 시)
PENDING → PENDING (다음 결재자로 이동)
```

---

### 3.3 기한 상태 조회

#### 함수: `get_deadline_status()`

**기능**: 결재의 기한 상태를 조회합니다.

**입력 파라미터**:

| 파라미터 | 타입 | 설명 |
|----------|------|------|
| approval | dict | 결재 데이터 |

**출력**: 상태 딕셔너리

**상태 유형**:
```python
# 반환 예시
{
    "status": "overdue",     # overdue, urgent, warning, normal
    "remaining_hours": -2.5, # 남은 시간 (음수는 초과)
    "deadline": "2026-01-04T14:00:00",
    "message": "기한 2.5시간 초과"
}
```

**임계값**:
```python
DEADLINE_WARNING_THRESHOLD = 2   # 2시간 전 경고
DEADLINE_URGENT_THRESHOLD = 1    # 1시간 전 긴급
```

---

### 3.4 알림 발송

#### 함수: `send_deadline_notification_email()`

**기능**: 기한 초과/임박 알림 이메일을 발송합니다.

**입력 파라미터**:

| 파라미터 | 타입 | 설명 |
|----------|------|------|
| approval | dict | 결재 데이터 |
| notification_type | str | 알림 유형 (overdue, urgent, warning) |

**출력**: 발송 성공 여부 (bool)

**처리 로직**:
1. 알림 유형별 메시지 생성
2. 현재 결재자 정보 조회
3. 이메일 내용 생성
4. 파일로 저장 (시뮬레이션)
5. 알림 이력 기록
6. 결과 반환

**저장 경로**: `/tmp/cs_approval_notifications/`

---

### 3.5 대시보드 요약

#### 함수: `get_approval_dashboard_summary()`

**기능**: 결재 현황 요약을 조회합니다.

**입력**: 없음

**출력**: 마크다운 형식 요약

**요약 내용**:
- 전체 결재 건수
- 대기 중 건수
- 승인 건수
- 반려 건수
- 유형별 통계

---

## 4. VLM 분석 기능

### 4.1 이미지 분석

#### 함수: `analyze_image()`

**기능**: VLM을 사용하여 결함 이미지를 분석합니다.

**입력 파라미터**:

| 파라미터 | 타입 | 설명 |
|----------|------|------|
| image | PIL.Image | 분석할 이미지 |

**출력**: 분석 결과 HTML

**처리 로직**:
1. 이미지 전처리
2. VLM 모델 추론
3. 결과 파싱
4. HTML 형식으로 변환
5. 결과 반환

**분석 결과 구조**:
```python
{
    "defect_type": "DEAD_PIXEL",
    "severity": "HIGH",
    "description": "화면 중앙부에 0.3mm 크기의 검은 점 발견",
    "cause": "TFT 소자 불량으로 추정",
    "recommendation": "해당 셀 교체 필요"
}
```

---

### 4.2 데모 분석

#### 함수: `analyze_image_demo()`

**기능**: 데모용 이미지 분석 (모델 없이 시뮬레이션)

**입력 파라미터**:

| 파라미터 | 타입 | 설명 |
|----------|------|------|
| image | PIL.Image | 분석할 이미지 |

**출력**: 분석 결과 HTML

**처리 로직**:
1. 이미지 존재 확인
2. 시뮬레이션 결과 생성
3. HTML 형식으로 변환
4. 결과 반환

---

## 5. GraphRAG 기능

### 5.1 지식베이스 초기화

#### 함수: `init_knowledge_base()`

**기능**: 지식그래프 기반 지식베이스를 초기화합니다.

**입력 파라미터**:

| 파라미터 | 타입 | 설명 |
|----------|------|------|
| use_neo4j | bool | Neo4j 사용 여부 |
| neo4j_password | str | Neo4j 비밀번호 |

**출력**: 초기화 결과 메시지

**처리 로직**:
1. 메모리 저장소 초기화
2. Neo4j 연결 (옵션)
3. 기본 노드/관계 생성
4. 초기화 완료 메시지 반환

---

### 5.2 노드 추가

#### 함수: `add_defect_node()`, `add_cause_node()`, `add_action_node()`

**기능**: 결함/원인/조치 노드를 추가합니다.

**Defect 노드 파라미터**:

| 파라미터 | 타입 | 설명 |
|----------|------|------|
| defect_id | str | 결함 ID |
| defect_type | str | 결함 유형 |
| korean_name | str | 한글명 |
| description | str | 설명 |
| severity | str | 심각도 |
| visual_char | str | 시각적 특성 |

**Cause 노드 파라미터**:

| 파라미터 | 타입 | 설명 |
|----------|------|------|
| cause_id | str | 원인 ID |
| cause_type | str | 원인 유형 |
| korean_name | str | 한글명 |
| description | str | 설명 |
| category | str | 카테고리 |

**Action 노드 파라미터**:

| 파라미터 | 타입 | 설명 |
|----------|------|------|
| action_id | str | 조치 ID |
| action_type | str | 조치 유형 |
| korean_name | str | 한글명 |
| description | str | 설명 |
| priority | str | 우선순위 |

---

### 5.3 관계 추가

#### 함수: `add_caused_by_relation()`, `add_requires_relation()`

**CAUSED_BY 관계 파라미터**:

| 파라미터 | 타입 | 설명 |
|----------|------|------|
| defect_id | str | 결함 ID |
| cause_id | str | 원인 ID |
| probability | float | 발생 확률 |
| evidence | str | 근거 |

**REQUIRES 관계 파라미터**:

| 파라미터 | 타입 | 설명 |
|----------|------|------|
| cause_id | str | 원인 ID |
| action_id | str | 조치 ID |
| effectiveness | float | 효과성 |

---

### 5.4 데이터 조회

#### 함수: `get_all_nodes()`, `get_all_relations()`, `query_defect_analysis()`

**get_all_nodes()**: 모든 노드 조회
**get_all_relations()**: 모든 관계 조회
**query_defect_analysis(defect_type)**: 특정 결함 관련 정보 조회

---

## 6. 빅데이터 분석 기능

### 6.1 파이프라인 초기화

#### 함수: `init_bigdata_pipeline()`

**기능**: 빅데이터 분석 파이프라인을 초기화합니다.

**처리 로직**:
1. Spark 세션 시뮬레이션
2. 데이터 소스 연결 설정
3. 파이프라인 설정 로드

---

### 6.2 분석 파이프라인 실행

#### 함수: `run_defect_analysis_pipeline()`

**입력 파라미터**:

| 파라미터 | 타입 | 설명 |
|----------|------|------|
| case_id | str | 케이스 ID |
| cell_id | str | 셀 ID |
| defect_type | str | 결함 유형 |
| defect_date | str | 발생 일자 |
| customer | str | 고객사 |
| severity | str | 심각도 |
| description | str | 상세 설명 |

**출력**: 분석 결과 HTML

**파이프라인 단계**:
1. 데이터 수집 (MES, 검사 데이터)
2. ETL 처리 (정제, 변환, 결합)
3. 분석/ML (패턴 분석, 이상 탐지)
4. 결과 저장 (데이터마트)

---

## 7. 대시보드 기능

### 7.1 품질 대시보드 생성

#### 함수: `generate_quality_dashboard()`

**기능**: 품질 현황 대시보드를 생성합니다.

**출력**: 대시보드 HTML

---

### 7.2 결함 차트 생성

#### 함수: `generate_defect_chart()`

**기능**: 결함 유형별 분포 차트를 생성합니다.

**출력**: Matplotlib Figure

---

### 7.3 설비 차트 생성

#### 함수: `generate_equipment_chart()`

**기능**: 설비별 불량률 차트를 생성합니다.

**출력**: Matplotlib Figure

---

### 7.4 고객 차트 생성

#### 함수: `generate_customer_chart()`

**기능**: 고객사별 품질 현황 차트를 생성합니다.

**출력**: Matplotlib Figure

---

## 8. 이메일/문서 생성 기능

### 8.1 고객 확인 이메일 생성

#### 함수: `generate_complaint_email()`

**입력 파라미터**:

| 파라미터 | 타입 | 설명 |
|----------|------|------|
| customer | str | 고객사 |
| product_model | str | 제품 모델 |
| lot_id | str | LOT ID |
| cell_id | str | CELL ID |
| defect_type | str | 결함 유형 |
| defect_description | str | 결함 설명 |
| severity | str | 심각도 |
| complaint_id | str | 불만접수 ID |

**출력**: (이메일 내용, 수신자 이메일, 수신자 이름)

---

### 8.2 이메일 번역

#### 함수: `translate_email()`

**입력 파라미터**:

| 파라미터 | 타입 | 설명 |
|----------|------|------|
| email_content | str | 이메일 내용 |
| target_language | str | 목표 언어 |

**출력**: 번역된 이메일 내용

**지원 언어**: 한국어, 영어, 일본어, 중국어

---

### 8.3 이메일 발송

#### 함수: `send_complaint_email()`

**입력 파라미터**:

| 파라미터 | 타입 | 설명 |
|----------|------|------|
| recipient_email | str | 수신자 이메일 |
| recipient_name | str | 수신자 이름 |
| email_content | str | 이메일 내용 |
| complaint_id | str | 불만접수 ID |

**출력**: 발송 결과 메시지

**저장 경로**: `/tmp/cs_emails/`

---

### 8.4 미팅 요청 이메일 생성

#### 함수: `generate_meeting_request_email()`

**출력**: (이메일 내용, 참석자 목록, 귀책부서, 미팅 일시)

---

### 8.5 1차 산출물 보고서 생성

#### 함수: `generate_first_output_report()`

**입력 파라미터**:

| 파라미터 | 타입 | 설명 |
|----------|------|------|
| complaint_id | str | 불만접수 ID |
| meeting_summary | str | 미팅 요약 |
| tasks_summary | str | 업무 할당 요약 |
| outputs_summary | str | 산출물 요약 |

**출력**: (결과 메시지, 보고서 경로)

**저장 경로**: `/tmp/cs_reports/first_output/`

---

### 8.6 2차 산출물 보고서 생성

#### 함수: `generate_second_output_report()`

**저장 경로**: `/tmp/cs_reports/second_output/`

---

### 8.7 고객 회신 이메일 생성

#### 함수: `generate_customer_reply_email()`

**입력 파라미터**:

| 파라미터 | 타입 | 설명 |
|----------|------|------|
| customer | str | 고객사 |
| complaint_id | str | 불만접수 ID |
| defect_type | str | 결함 유형 |
| report_path | str | 보고서 경로 |

**출력**: (이메일 내용, 수신자 이메일, 수신자 이름)

---

## 9. API 레퍼런스

### 9.1 CS 워크플로우 API

| 함수명 | 설명 |
|--------|------|
| `init_cs_workflow()` | 워크플로우 초기화 |
| `create_cs_complaint()` | 불만 접수 생성 |
| `perform_first_analysis()` | 1차 분석 실행 |
| `perform_second_analysis()` | 2차 분석 실행 |
| `generate_final_report()` | 최종 보고서 생성 |
| `get_complaints_list()` | 불만 목록 조회 |

### 9.2 결재 시스템 API

| 함수명 | 설명 |
|--------|------|
| `create_approval_request()` | 결재 요청 생성 |
| `get_approval_status()` | 결재 상태 조회 |
| `process_approval()` | 결재 처리 |
| `get_pending_approvals()` | 대기 결재 조회 |
| `get_deadline_status()` | 기한 상태 조회 |
| `send_deadline_notification_email()` | 알림 발송 |

### 9.3 GraphRAG API

| 함수명 | 설명 |
|--------|------|
| `init_knowledge_base()` | 지식베이스 초기화 |
| `add_defect_node()` | 결함 노드 추가 |
| `add_cause_node()` | 원인 노드 추가 |
| `add_action_node()` | 조치 노드 추가 |
| `add_caused_by_relation()` | CAUSED_BY 관계 추가 |
| `add_requires_relation()` | REQUIRES 관계 추가 |
| `get_all_nodes()` | 전체 노드 조회 |
| `query_defect_analysis()` | 결함 분석 조회 |

### 9.4 시각화 API

| 함수명 | 설명 |
|--------|------|
| `create_graph_visualization()` | 전체 그래프 시각화 |
| `create_subgraph_visualization()` | 서브그래프 시각화 |
| `generate_defect_chart()` | 결함 차트 생성 |
| `generate_equipment_chart()` | 설비 차트 생성 |
| `generate_customer_chart()` | 고객 차트 생성 |

### 9.5 이메일/문서 API

| 함수명 | 설명 |
|--------|------|
| `generate_complaint_email()` | 고객 이메일 생성 |
| `translate_email()` | 이메일 번역 |
| `send_complaint_email()` | 이메일 발송 |
| `generate_meeting_request_email()` | 미팅 요청 생성 |
| `generate_first_output_report()` | 1차 보고서 생성 |
| `generate_second_output_report()` | 2차 보고서 생성 |
| `generate_customer_reply_email()` | 회신 이메일 생성 |

---

## 부록: 상수 정의

### 결재 상태

```python
APPROVAL_STATUS = {
    "PENDING": "대기",
    "APPROVED": "승인",
    "REJECTED": "반려",
    "CANCELLED": "취소"
}
```

### 결재 유형

```python
APPROVAL_TYPES = {
    "COMPLAINT_EMAIL": "고객 확인 이메일",
    "MEETING_REQUEST": "미팅 요청",
    "FIRST_ANALYSIS": "1차 분석 결과",
    "TASK_ASSIGNMENT": "업무 할당",
    "FINAL_REPORT": "최종 보고서",
    "CUSTOMER_REPLY": "고객 회신",
}
```

### 결함 유형

```python
DEFECT_TYPES = {
    "DEAD_PIXEL": "데드 픽셀",
    "BRIGHT_SPOT": "브라이트 스팟",
    "LINE_DEFECT": "라인 결함",
    "MURA": "무라",
    "SCRATCH": "스크래치",
    "TOUCH_FAIL": "터치 불량",
}
```

### 심각도

```python
SEVERITY_LEVELS = {
    "LOW": "낮음",
    "MEDIUM": "보통",
    "HIGH": "높음",
    "CRITICAL": "심각",
}
```

---

*본 문서는 SDC Customer Quality System v2.0 기준으로 작성되었습니다.*
