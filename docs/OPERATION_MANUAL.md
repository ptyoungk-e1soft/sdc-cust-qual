# SDC Customer Quality System - 운영 매뉴얼

**버전**: 2.0
**최종 수정일**: 2026-01-04
**작성자**: 품질관리시스템개발팀

---

## 목차

1. [시스템 요구사항](#1-시스템-요구사항)
2. [설치 및 설정](#2-설치-및-설정)
3. [시스템 시작/중지](#3-시스템-시작중지)
4. [설정 관리](#4-설정-관리)
5. [결재 시스템 관리](#5-결재-시스템-관리)
6. [데이터 관리](#6-데이터-관리)
7. [로그 관리](#7-로그-관리)
8. [백업 및 복구](#8-백업-및-복구)
9. [모니터링](#9-모니터링)
10. [트러블슈팅](#10-트러블슈팅)
11. [보안 관리](#11-보안-관리)

---

## 1. 시스템 요구사항

### 1.1 하드웨어 요구사항

| 구분 | 최소 사양 | 권장 사양 |
|------|----------|----------|
| CPU | 8코어 | 16코어 이상 |
| RAM | 32GB | 64GB 이상 |
| GPU | NVIDIA RTX 3080 (10GB) | NVIDIA RTX 4090 (24GB) |
| Storage | 500GB SSD | 1TB NVMe SSD |
| Network | 1Gbps | 10Gbps |

### 1.2 소프트웨어 요구사항

| 소프트웨어 | 버전 | 비고 |
|-----------|------|------|
| OS | Ubuntu 22.04 LTS | Linux 권장 |
| Python | 3.10+ | 3.11 권장 |
| CUDA | 12.0+ | GPU 사용 시 필수 |
| Docker | 24.0+ | 선택사항 |
| Neo4j | 5.x | GraphRAG 사용 시 필수 |

### 1.3 Python 패키지

```txt
gradio>=4.0.0
torch>=2.0.0
transformers>=4.36.0
peft>=0.7.0
neo4j>=5.0.0
pandas>=2.0.0
networkx>=3.0
matplotlib>=3.7.0
python-docx>=0.8.11
openpyxl>=3.1.0
Pillow>=10.0.0
requests>=2.31.0
```

---

## 2. 설치 및 설정

### 2.1 환경 설정

```bash
# 프로젝트 디렉토리 이동
cd /home/ptyoung/sdc-cust-qual

# 가상환경 생성
python3 -m venv .venv

# 가상환경 활성화
source .venv/bin/activate

# 패키지 설치
pip install -r requirements.txt
```

### 2.2 Neo4j 설정

#### Docker를 사용한 설치

```bash
# Neo4j 컨테이너 실행
docker run -d \
  --name neo4j \
  -p 7474:7474 \
  -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  -v neo4j_data:/data \
  neo4j:5
```

#### 접속 정보

| 항목 | 값 |
|------|-----|
| Browser URL | http://localhost:7474 |
| Bolt URL | bolt://localhost:7687 |
| Username | neo4j |
| Password | password |

### 2.3 GPU 설정

```bash
# CUDA 확인
nvidia-smi

# PyTorch GPU 확인
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### 2.4 디렉토리 구조

```
/home/ptyoung/sdc-cust-qual/
├── demo/
│   └── app_full.py          # 메인 애플리케이션
├── docs/                     # 문서
├── data/                     # 데이터 파일
├── configs/                  # 설정 파일
├── output/                   # 모델 출력
└── .venv/                    # 가상환경
```

### 2.5 런타임 디렉토리

```
/tmp/
├── cs_emails/                    # 이메일 로그
├── cs_approval_notifications/    # 알림 로그
├── cs_reports/                   # 보고서
│   ├── first_output/
│   ├── second_output/
│   └── final/
└── mockdata/                     # 목업 데이터
```

---

## 3. 시스템 시작/중지

### 3.1 시스템 시작

```bash
# 가상환경 활성화
source .venv/bin/activate

# 데모 시작 (포그라운드)
python3 demo/app_full.py

# 데모 시작 (백그라운드)
nohup python3 demo/app_full.py > /tmp/demo.log 2>&1 &
```

### 3.2 시스템 상태 확인

```bash
# 프로세스 확인
pgrep -af "app_full.py"

# 포트 확인
ss -tlnp | grep 7860

# 로그 확인
tail -f /tmp/demo.log
```

### 3.3 시스템 중지

```bash
# 프로세스 종료
pkill -f "python.*app_full.py"

# 또는 PID로 종료
kill -9 <PID>
```

### 3.4 시스템 재시작

```bash
# 중지 후 시작
pkill -f "python.*app_full.py"
sleep 2
source .venv/bin/activate
nohup python3 demo/app_full.py > /tmp/demo.log 2>&1 &
```

### 3.5 서비스 등록 (systemd)

```ini
# /etc/systemd/system/sdc-quality.service
[Unit]
Description=SDC Customer Quality System
After=network.target

[Service]
Type=simple
User=ptyoung
WorkingDirectory=/home/ptyoung/sdc-cust-qual
ExecStart=/home/ptyoung/sdc-cust-qual/.venv/bin/python3 demo/app_full.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# 서비스 등록 및 시작
sudo systemctl daemon-reload
sudo systemctl enable sdc-quality
sudo systemctl start sdc-quality
```

---

## 4. 설정 관리

### 4.1 서버 포트 설정

`app_full.py` 파일 내 포트 설정:

```python
PORT = 7860  # 기본 포트
```

### 4.2 결재 라인 설정

```python
APPROVAL_LINE = {
    "COMPLAINT_EMAIL": [
        {"position": "팀장", "name": "이정호", "email": "jhlee@sdc.com", "dept": "품질관리팀"},
    ],
    "FIRST_ANALYSIS": [
        {"position": "팀장", "name": "이정호", "email": "jhlee@sdc.com", "dept": "품질관리팀"},
        {"position": "부장", "name": "김대영", "email": "dykim@sdc.com", "dept": "품질본부"},
    ],
    "FINAL_REPORT": [
        {"position": "팀장", "name": "이정호", "email": "jhlee@sdc.com", "dept": "품질관리팀"},
        {"position": "부장", "name": "김대영", "email": "dykim@sdc.com", "dept": "품질본부"},
        {"position": "상무", "name": "박철수", "email": "cspark@sdc.com", "dept": "품질담당"},
    ],
    # ... 기타 유형
}
```

### 4.3 결재 기한 설정

```python
APPROVAL_DEADLINES = {
    "COMPLAINT_EMAIL": 4,      # 4시간 이내
    "MEETING_REQUEST": 8,      # 8시간 이내
    "FIRST_ANALYSIS": 24,      # 24시간 이내
    "TASK_ASSIGNMENT": 8,      # 8시간 이내
    "FINAL_REPORT": 48,        # 48시간 이내
    "CUSTOMER_REPLY": 24,      # 24시간 이내
}
```

### 4.4 알림 임계값 설정

```python
DEADLINE_WARNING_THRESHOLD = 2   # 2시간 전 경고
DEADLINE_URGENT_THRESHOLD = 1    # 1시간 전 긴급
```

### 4.5 고객사 연락처 설정

```python
CUSTOMER_CONTACTS = {
    "APPLE": {"email": "quality@apple.com", "name": "John Smith", "region": "US"},
    "SAMSUNG_MOBILE": {"email": "quality@samsung.com", "name": "김철수", "region": "KR"},
    # ... 기타 고객사
}
```

### 4.6 내부 부서 설정

```python
INTERNAL_DEPARTMENTS = {
    "TFT공정": {"email": "tft@sdc.com", "manager": "박영수", "ext": "1234"},
    "CF공정": {"email": "cf@sdc.com", "manager": "이민정", "ext": "1235"},
    "OLED공정": {"email": "oled@sdc.com", "manager": "최대호", "ext": "1236"},
    # ... 기타 부서
}
```

---

## 5. 결재 시스템 관리

### 5.1 결재 상태

| 상태 | 코드 | 설명 |
|------|------|------|
| 대기 | PENDING | 결재 대기 중 |
| 승인 | APPROVED | 결재 승인됨 |
| 반려 | REJECTED | 결재 반려됨 |
| 취소 | CANCELLED | 결재 취소됨 |

### 5.2 결재 유형

| 유형 | 코드 | 기한 |
|------|------|------|
| 고객 이메일 | COMPLAINT_EMAIL | 4시간 |
| 미팅 요청 | MEETING_REQUEST | 8시간 |
| 1차 분석 | FIRST_ANALYSIS | 24시간 |
| 업무 할당 | TASK_ASSIGNMENT | 8시간 |
| 최종 보고서 | FINAL_REPORT | 48시간 |
| 고객 회신 | CUSTOMER_REPLY | 24시간 |

### 5.3 결재 데이터 저장 위치

```python
# 메모리 기반 저장소 (런타임)
approval_storage = {}        # 결재 데이터
notification_history = {}    # 알림 이력
```

### 5.4 결재 로그 위치

```
/tmp/cs_approval_logs/
├── APR-20260104-001.json    # 결재별 로그
├── APR-20260104-002.json
└── ...
```

### 5.5 알림 로그 위치

```
/tmp/cs_approval_notifications/
├── notification_APR-20260104-001_20260104_120000.txt
└── ...
```

---

## 6. 데이터 관리

### 6.1 불만 접수 데이터

```python
# 저장 구조
cs_workflow_storage = {
    "complaints": {},        # 불만 접수
    "first_analyses": {},    # 1차 분석
    "second_analyses": {},   # 2차 분석
    "reports": {},           # 보고서
    "outputs": {},           # 산출물
}
```

### 6.2 보고서 저장 위치

```
/tmp/cs_reports/
├── first_output/
│   └── CS-20260104-001_first_output_report.docx
├── second_output/
│   └── CS-20260104-001_second_output_report.docx
└── final/
    └── CS-20260104-001_final_report.docx
```

### 6.3 이메일 로그 위치

```
/tmp/cs_emails/
├── CS-20260104-001_customer_email_20260104_100000.txt
├── CS-20260104-001_meeting_request_20260104_101500.txt
└── ...
```

### 6.4 목업 데이터 위치

```
/tmp/mockdata/
├── development/
│   └── dev_data.json
├── manufacturing/
│   └── mfg_data.json
└── mes/
    └── mes_data.json
```

### 6.5 데이터 정리

```bash
# 임시 파일 정리
rm -rf /tmp/cs_emails/*
rm -rf /tmp/cs_reports/*
rm -rf /tmp/cs_approval_*
rm -rf /tmp/mockdata/*
```

---

## 7. 로그 관리

### 7.1 애플리케이션 로그

```bash
# 로그 파일 위치
/tmp/demo.log

# 실시간 모니터링
tail -f /tmp/demo.log

# 최근 100줄 확인
tail -100 /tmp/demo.log
```

### 7.2 로그 형식

```
[2026-01-04 10:00:00] INFO: 시스템 시작
[2026-01-04 10:00:05] INFO: 포트 7860에서 서비스 시작
[2026-01-04 10:01:00] INFO: 불만 접수 생성 - CS-20260104-001
```

### 7.3 로그 로테이션

```bash
# logrotate 설정 (/etc/logrotate.d/sdc-quality)
/tmp/demo.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
}
```

### 7.4 로그 분석

```bash
# 에러 로그 추출
grep -i "error\|exception" /tmp/demo.log

# 특정 날짜 로그
grep "2026-01-04" /tmp/demo.log

# 결재 관련 로그
grep "결재" /tmp/demo.log
```

---

## 8. 백업 및 복구

### 8.1 백업 대상

| 대상 | 위치 | 빈도 |
|------|------|------|
| 애플리케이션 | /home/ptyoung/sdc-cust-qual | 일일 |
| 보고서 | /tmp/cs_reports | 일일 |
| 이메일 로그 | /tmp/cs_emails | 주간 |
| Neo4j 데이터 | Docker volume | 일일 |

### 8.2 백업 스크립트

```bash
#!/bin/bash
# /opt/scripts/backup.sh

BACKUP_DIR="/backup/sdc-quality"
DATE=$(date +%Y%m%d)

# 애플리케이션 백업
tar -czf $BACKUP_DIR/app_$DATE.tar.gz /home/ptyoung/sdc-cust-qual

# 보고서 백업
tar -czf $BACKUP_DIR/reports_$DATE.tar.gz /tmp/cs_reports

# Neo4j 백업
docker exec neo4j neo4j-admin database dump neo4j --to-path=/tmp
docker cp neo4j:/tmp/neo4j.dump $BACKUP_DIR/neo4j_$DATE.dump

# 오래된 백업 삭제 (30일 이상)
find $BACKUP_DIR -mtime +30 -delete
```

### 8.3 복구 절차

```bash
# 애플리케이션 복구
cd /home/ptyoung
tar -xzf /backup/sdc-quality/app_20260104.tar.gz

# 보고서 복구
tar -xzf /backup/sdc-quality/reports_20260104.tar.gz -C /tmp

# Neo4j 복구
docker cp /backup/sdc-quality/neo4j_20260104.dump neo4j:/tmp/
docker exec neo4j neo4j-admin database load neo4j --from-path=/tmp
```

---

## 9. 모니터링

### 9.1 시스템 상태 확인

```bash
# 프로세스 상태
ps aux | grep app_full.py

# 메모리 사용량
free -h

# 디스크 사용량
df -h

# GPU 상태
nvidia-smi
```

### 9.2 서비스 상태 확인

```bash
# 포트 확인
ss -tlnp | grep 7860

# HTTP 응답 확인
curl -s -o /dev/null -w "%{http_code}" http://localhost:7860

# Neo4j 상태
curl -s http://localhost:7474
```

### 9.3 모니터링 스크립트

```bash
#!/bin/bash
# /opt/scripts/health_check.sh

# Gradio 서비스 확인
if ! curl -s http://localhost:7860 > /dev/null; then
    echo "ALERT: Gradio service is down!"
    # 알림 발송 로직
fi

# Neo4j 확인
if ! curl -s http://localhost:7474 > /dev/null; then
    echo "ALERT: Neo4j is down!"
fi

# GPU 메모리 확인
GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [ "$GPU_MEM" -gt 20000 ]; then
    echo "WARNING: GPU memory usage is high: ${GPU_MEM}MB"
fi
```

### 9.4 Cron 설정

```bash
# crontab -e
# 5분마다 헬스체크
*/5 * * * * /opt/scripts/health_check.sh >> /var/log/health_check.log 2>&1

# 매일 02시 백업
0 2 * * * /opt/scripts/backup.sh >> /var/log/backup.log 2>&1
```

---

## 10. 트러블슈팅

### 10.1 서비스 시작 실패

**증상**: 포트 7860이 열리지 않음

**해결방법**:
```bash
# 기존 프로세스 확인
lsof -i :7860

# 강제 종료
fuser -k 7860/tcp

# 재시작
python3 demo/app_full.py
```

### 10.2 GPU 메모리 부족

**증상**: CUDA out of memory 오류

**해결방법**:
```bash
# GPU 메모리 정리
nvidia-smi --gpu-reset

# 또는 프로세스 종료
pkill -f python

# 메모리 확인 후 재시작
nvidia-smi
python3 demo/app_full.py
```

### 10.3 Neo4j 연결 실패

**증상**: Neo4j connection refused

**해결방법**:
```bash
# Neo4j 상태 확인
docker ps | grep neo4j

# Neo4j 재시작
docker restart neo4j

# 로그 확인
docker logs neo4j
```

### 10.4 보고서 생성 실패

**증상**: Word 보고서가 생성되지 않음

**해결방법**:
```bash
# 디렉토리 권한 확인
ls -la /tmp/cs_reports/

# 디렉토리 생성
mkdir -p /tmp/cs_reports/{first_output,second_output,final}

# 권한 설정
chmod 777 /tmp/cs_reports
```

### 10.5 이메일 발송 실패

**증상**: 이메일 로그가 생성되지 않음

**해결방법**:
```bash
# 디렉토리 확인
mkdir -p /tmp/cs_emails

# 권한 확인
chmod 777 /tmp/cs_emails
```

### 10.6 한글 깨짐

**증상**: 그래프나 보고서에서 한글이 깨짐

**해결방법**:
```bash
# 한글 폰트 설치
sudo apt-get install fonts-nanum

# 폰트 캐시 갱신
fc-cache -fv

# 애플리케이션 재시작
```

---

## 11. 보안 관리

### 11.1 접근 제어

- 시스템은 localhost에서만 접근 가능 (기본 설정)
- 외부 접근 시 방화벽/리버스 프록시 설정 필요

### 11.2 인증 설정 (Gradio)

```python
# 기본 인증 추가 시
demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    auth=("admin", "password")  # 인증 추가
)
```

### 11.3 Neo4j 보안

```bash
# 비밀번호 변경
docker exec -it neo4j cypher-shell
> CALL dbms.security.changePassword('new_password');
```

### 11.4 로그 보안

- 민감 정보가 포함된 로그는 정기적으로 삭제
- 로그 파일 접근 권한 제한

```bash
chmod 600 /tmp/demo.log
chmod 600 /tmp/cs_emails/*
```

### 11.5 데이터 보안

- 고객 정보는 암호화 저장 권장
- 정기적인 데이터 백업
- 불필요한 데이터 정기 삭제

---

## 부록: 주요 명령어 정리

```bash
# 시스템 시작
source .venv/bin/activate && python3 demo/app_full.py

# 백그라운드 시작
nohup python3 demo/app_full.py > /tmp/demo.log 2>&1 &

# 시스템 중지
pkill -f "python.*app_full.py"

# 상태 확인
pgrep -af "app_full.py"
ss -tlnp | grep 7860

# 로그 확인
tail -f /tmp/demo.log

# Neo4j 시작
docker start neo4j

# Neo4j 중지
docker stop neo4j

# 임시 파일 정리
rm -rf /tmp/cs_*
```

---

*본 문서는 SDC Customer Quality System v2.0 운영 기준으로 작성되었습니다.*
