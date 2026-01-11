"""
CS 불만 접수 관리 모듈
고객 불량 접수부터 분석 완료까지의 워크플로우 관리
"""

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Any, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ComplaintStatus(Enum):
    """불만 처리 상태"""
    RECEIVED = "접수"
    FIRST_ANALYSIS = "1차분석중"
    DEPT_ASSIGNED = "귀책부서배정"
    SECOND_ANALYSIS = "2차분석중"
    REPORT_WRITING = "보고서작성중"
    COMPLETED = "완료"
    CLOSED = "종결"


class DefectCategory(Enum):
    """결함 유형 분류"""
    DISPLAY_DEFECT = "디스플레이결함"
    TOUCH_DEFECT = "터치결함"
    POWER_DEFECT = "전원결함"
    STRUCTURAL_DEFECT = "구조결함"
    OTHER = "기타"


class ResponsibleDept(Enum):
    """귀책 부서"""
    TFT = "TFT공정"
    CF = "CF공정"
    CELL = "CELL공정"
    MODULE = "모듈공정"
    MATERIAL = "자재"
    EQUIPMENT = "설비"
    DESIGN = "설계"
    UNKNOWN = "미정"


@dataclass
class CSComplaint:
    """CS 불만 접수 데이터"""
    complaint_id: str
    receipt_date: datetime
    customer: str
    product_model: str
    lot_id: str
    cell_id: str
    defect_type: str
    defect_description: str
    defect_image_path: str = ""
    severity: str = "MEDIUM"
    status: str = "접수"
    responsible_dept: str = "미정"

    # 분석 결과
    first_analysis_result: Dict[str, Any] = field(default_factory=dict)
    second_analysis_result: Dict[str, Any] = field(default_factory=dict)
    root_cause: str = ""
    countermeasure: str = ""

    # 연관 데이터
    related_lots: List[str] = field(default_factory=list)
    related_cases: List[str] = field(default_factory=list)

    # 타임라인
    first_analysis_date: Optional[datetime] = None
    dept_assign_date: Optional[datetime] = None
    second_analysis_date: Optional[datetime] = None
    completion_date: Optional[datetime] = None

    # 담당자
    cs_handler: str = ""
    quality_analyst: str = ""
    dept_analyst: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        result = asdict(self)
        # datetime 변환
        for key, value in result.items():
            if isinstance(value, datetime):
                result[key] = value.isoformat()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CSComplaint":
        """딕셔너리에서 생성"""
        # datetime 파싱
        date_fields = ['receipt_date', 'first_analysis_date', 'dept_assign_date',
                       'second_analysis_date', 'completion_date']
        for field in date_fields:
            if data.get(field) and isinstance(data[field], str):
                try:
                    data[field] = datetime.fromisoformat(data[field])
                except:
                    data[field] = None
        return cls(**data)


class CSComplaintManager:
    """CS 불만 관리자"""

    def __init__(self, data_dir: str = "/tmp/cs_complaints"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.complaints: Dict[str, CSComplaint] = {}
        self._load_complaints()

    def _load_complaints(self):
        """저장된 불만 데이터 로드"""
        complaints_file = os.path.join(self.data_dir, "complaints.json")
        if os.path.exists(complaints_file):
            with open(complaints_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    complaint = CSComplaint.from_dict(item)
                    self.complaints[complaint.complaint_id] = complaint

    def _save_complaints(self):
        """불만 데이터 저장"""
        complaints_file = os.path.join(self.data_dir, "complaints.json")
        data = [c.to_dict() for c in self.complaints.values()]
        with open(complaints_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    def create_complaint(self,
                         customer: str,
                         product_model: str,
                         lot_id: str,
                         cell_id: str,
                         defect_type: str,
                         defect_description: str,
                         defect_image_path: str = "",
                         severity: str = "MEDIUM") -> CSComplaint:
        """새 불만 접수 생성"""
        complaint_id = f"CS{datetime.now().strftime('%Y%m%d%H%M%S')}"

        complaint = CSComplaint(
            complaint_id=complaint_id,
            receipt_date=datetime.now(),
            customer=customer,
            product_model=product_model,
            lot_id=lot_id,
            cell_id=cell_id,
            defect_type=defect_type,
            defect_description=defect_description,
            defect_image_path=defect_image_path,
            severity=severity,
            status=ComplaintStatus.RECEIVED.value,
            cs_handler=f"CS_Handler_{datetime.now().strftime('%H%M')}"
        )

        self.complaints[complaint_id] = complaint
        self._save_complaints()

        logger.info(f"새 불만 접수: {complaint_id}")
        return complaint

    def update_status(self, complaint_id: str, new_status: ComplaintStatus):
        """상태 업데이트"""
        if complaint_id in self.complaints:
            self.complaints[complaint_id].status = new_status.value
            self._save_complaints()

    def assign_responsible_dept(self, complaint_id: str, dept: ResponsibleDept, analyst: str = ""):
        """귀책 부서 배정"""
        if complaint_id in self.complaints:
            complaint = self.complaints[complaint_id]
            complaint.responsible_dept = dept.value
            complaint.dept_assign_date = datetime.now()
            complaint.status = ComplaintStatus.DEPT_ASSIGNED.value
            if analyst:
                complaint.dept_analyst = analyst
            self._save_complaints()

    def update_first_analysis(self, complaint_id: str, result: Dict[str, Any], analyst: str = ""):
        """1차 분석 결과 업데이트"""
        if complaint_id in self.complaints:
            complaint = self.complaints[complaint_id]
            complaint.first_analysis_result = result
            complaint.first_analysis_date = datetime.now()
            complaint.status = ComplaintStatus.FIRST_ANALYSIS.value
            if analyst:
                complaint.quality_analyst = analyst
            self._save_complaints()

    def update_second_analysis(self, complaint_id: str, result: Dict[str, Any],
                               root_cause: str = "", countermeasure: str = ""):
        """2차 분석 결과 업데이트"""
        if complaint_id in self.complaints:
            complaint = self.complaints[complaint_id]
            complaint.second_analysis_result = result
            complaint.second_analysis_date = datetime.now()
            complaint.status = ComplaintStatus.SECOND_ANALYSIS.value
            if root_cause:
                complaint.root_cause = root_cause
            if countermeasure:
                complaint.countermeasure = countermeasure
            self._save_complaints()

    def complete_complaint(self, complaint_id: str):
        """불만 처리 완료"""
        if complaint_id in self.complaints:
            complaint = self.complaints[complaint_id]
            complaint.completion_date = datetime.now()
            complaint.status = ComplaintStatus.COMPLETED.value
            self._save_complaints()

    def get_complaint(self, complaint_id: str) -> Optional[CSComplaint]:
        """불만 조회"""
        return self.complaints.get(complaint_id)

    def get_all_complaints(self) -> List[CSComplaint]:
        """전체 불만 목록"""
        return list(self.complaints.values())

    def get_complaints_by_status(self, status: ComplaintStatus) -> List[CSComplaint]:
        """상태별 불만 조회"""
        return [c for c in self.complaints.values() if c.status == status.value]

    def get_complaints_by_customer(self, customer: str) -> List[CSComplaint]:
        """고객사별 불만 조회"""
        return [c for c in self.complaints.values() if c.customer == customer]

    def generate_sample_complaints(self, count: int = 10) -> List[CSComplaint]:
        """샘플 불만 데이터 생성"""
        import random

        customers = ["APPLE", "SAMSUNG_MOBILE", "LG_MOBILE", "GOOGLE", "XIAOMI", "HUAWEI", "META"]
        products = ["OLED_67_FHD", "OLED_61_FHD", "LTPO_68_QHD", "LCD_109_2K", "OLED_76_FOLD"]
        defect_types = ["DEAD_PIXEL", "BRIGHT_SPOT", "LINE_DEFECT", "MURA", "SCRATCH", "TOUCH_FAIL"]
        severities = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

        defect_descriptions = {
            "DEAD_PIXEL": "화면 중앙부에 검은색 점 발견. 크기 약 0.3mm",
            "BRIGHT_SPOT": "좌측 상단 영역에 밝은 점 다수 발생. 백색 배경에서 육안 확인",
            "LINE_DEFECT": "수직 방향 라인 결함. 화면 전체 영역에 걸쳐 발생",
            "MURA": "특정 그레이 레벨에서 얼룩 현상 발생. 중앙부 집중",
            "SCRATCH": "표면 스크래치 발견. 길이 약 15mm, 깊이 얕음",
            "TOUCH_FAIL": "특정 영역 터치 무반응. 우측 하단 약 20% 영역"
        }

        complaints = []
        base_date = datetime.now()

        for i in range(count):
            defect = random.choice(defect_types)
            receipt_date = base_date.replace(
                day=max(1, base_date.day - random.randint(0, 30)),
                hour=random.randint(8, 18)
            )

            complaint = CSComplaint(
                complaint_id=f"CS{receipt_date.strftime('%Y%m%d')}{i:04d}",
                receipt_date=receipt_date,
                customer=random.choice(customers),
                product_model=random.choice(products),
                lot_id=f"LOT{receipt_date.strftime('%Y%m%d')}{random.randint(100, 999)}",
                cell_id=f"CELL{random.randint(10000, 99999)}",
                defect_type=defect,
                defect_description=defect_descriptions.get(defect, "결함 발생"),
                defect_image_path=f"/data/defect_images/{defect.lower()}_{i:04d}.png",
                severity=random.choice(severities),
                status=random.choice([s.value for s in ComplaintStatus]),
                responsible_dept=random.choice([d.value for d in ResponsibleDept]),
                cs_handler=f"CS_Handler_{random.randint(1, 10):02d}",
                quality_analyst=f"QA_Analyst_{random.randint(1, 5):02d}",
                dept_analyst=f"Dept_Analyst_{random.randint(1, 8):02d}"
            )

            self.complaints[complaint.complaint_id] = complaint
            complaints.append(complaint)

        self._save_complaints()
        logger.info(f"샘플 불만 데이터 {count}건 생성 완료")
        return complaints
