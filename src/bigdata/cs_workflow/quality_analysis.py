"""
품질 분석 모듈
1차 기본 분석 및 2차 상세 분석 처리
"""

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Any, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class AnalysisType(Enum):
    """분석 유형"""
    FIRST_BASIC = "1차기본분석"
    SECOND_DETAILED = "2차상세분석"


class AnalysisMethod(Enum):
    """분석 방법"""
    BIGDATA = "빅데이터분석"
    IMAGE = "이미지분석"
    GRAPHRAG = "GraphRAG분석"
    PAST_CASE = "과거사례분석"


@dataclass
class AnalysisResult:
    """분석 결과"""
    analysis_id: str
    complaint_id: str
    analysis_type: str
    analysis_date: datetime
    analyst: str

    # 분석 방법별 결과
    bigdata_result: Dict[str, Any] = field(default_factory=dict)
    image_result: Dict[str, Any] = field(default_factory=dict)
    graphrag_result: Dict[str, Any] = field(default_factory=dict)
    past_case_result: Dict[str, Any] = field(default_factory=dict)

    # 종합 결론
    defect_classification: str = ""
    root_cause: str = ""
    responsible_dept: str = ""
    confidence_score: float = 0.0

    # 권장 대책
    countermeasures: List[str] = field(default_factory=list)
    prevention_measures: List[str] = field(default_factory=list)

    # 관련 데이터
    related_lots: List[str] = field(default_factory=list)
    related_equipment: List[str] = field(default_factory=list)
    similar_cases: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        result = asdict(self)
        if isinstance(result.get('analysis_date'), datetime):
            result['analysis_date'] = result['analysis_date'].isoformat()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnalysisResult":
        """딕셔너리에서 생성"""
        if data.get('analysis_date') and isinstance(data['analysis_date'], str):
            try:
                data['analysis_date'] = datetime.fromisoformat(data['analysis_date'])
            except:
                data['analysis_date'] = datetime.now()
        return cls(**data)


class QualityAnalyzer:
    """품질 분석기"""

    def __init__(self, data_dir: str = "/tmp/quality_analysis"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self.results: Dict[str, AnalysisResult] = {}
        self._load_results()

        # 빅데이터 마트 경로
        self.datamart_dir = "/tmp/datamart_integrated"

    def _load_results(self):
        """저장된 분석 결과 로드"""
        results_file = os.path.join(self.data_dir, "analysis_results.json")
        if os.path.exists(results_file):
            with open(results_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    result = AnalysisResult.from_dict(item)
                    self.results[result.analysis_id] = result

    def _save_results(self):
        """분석 결과 저장"""
        results_file = os.path.join(self.data_dir, "analysis_results.json")
        data = [r.to_dict() for r in self.results.values()]
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    def perform_first_analysis(self,
                               complaint_id: str,
                               defect_type: str,
                               lot_id: str,
                               cell_id: str,
                               product_model: str,
                               analyst: str = "QA_System") -> AnalysisResult:
        """
        1차 기본 분석 수행
        - 빅데이터 기반 연관 분석
        - 결함 유형 분류
        - 귀책 부서 추정
        """
        analysis_id = f"AN1_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        # 1. 빅데이터 분석
        bigdata_result = self._analyze_bigdata(defect_type, lot_id, product_model)

        # 2. 귀책 부서 추정
        responsible_dept = self._estimate_responsible_dept(defect_type, bigdata_result)

        # 3. 유사 케이스 검색
        similar_cases = self._find_similar_cases(defect_type, product_model)

        result = AnalysisResult(
            analysis_id=analysis_id,
            complaint_id=complaint_id,
            analysis_type=AnalysisType.FIRST_BASIC.value,
            analysis_date=datetime.now(),
            analyst=analyst,
            bigdata_result=bigdata_result,
            defect_classification=defect_type,
            responsible_dept=responsible_dept,
            confidence_score=bigdata_result.get("confidence", 0.75),
            related_lots=bigdata_result.get("related_lots", []),
            related_equipment=bigdata_result.get("related_equipment", []),
            similar_cases=similar_cases
        )

        self.results[analysis_id] = result
        self._save_results()

        logger.info(f"1차 분석 완료: {analysis_id}, 귀책부서: {responsible_dept}")
        return result

    def perform_second_analysis(self,
                                complaint_id: str,
                                first_analysis_id: str,
                                defect_image_path: str = "",
                                analyst: str = "Dept_Analyst") -> AnalysisResult:
        """
        2차 상세 분석 수행
        - 1차 분석 결과 기반
        - 결함 이미지 분석
        - GraphRAG 기반 원인/대책 분석
        - 과거 사례 분석
        """
        analysis_id = f"AN2_{datetime.now().strftime('%Y%m%d%H%M%S')}"

        # 1차 분석 결과 가져오기
        first_result = self.results.get(first_analysis_id)
        if not first_result:
            raise ValueError(f"1차 분석 결과를 찾을 수 없습니다: {first_analysis_id}")

        # 1. 이미지 분석 (Cosmos VLM 연동 가정)
        image_result = self._analyze_defect_image(defect_image_path)

        # 2. GraphRAG 분석
        graphrag_result = self._analyze_with_graphrag(
            first_result.defect_classification,
            first_result.bigdata_result
        )

        # 3. 과거 사례 분석
        past_case_result = self._analyze_past_cases(
            first_result.defect_classification,
            first_result.similar_cases
        )

        # 4. 종합 분석 및 대책 수립
        root_cause = self._determine_root_cause(
            first_result.bigdata_result,
            image_result,
            graphrag_result,
            past_case_result
        )

        countermeasures = self._generate_countermeasures(
            root_cause,
            graphrag_result,
            past_case_result
        )

        prevention_measures = self._generate_prevention_measures(
            root_cause,
            graphrag_result
        )

        result = AnalysisResult(
            analysis_id=analysis_id,
            complaint_id=complaint_id,
            analysis_type=AnalysisType.SECOND_DETAILED.value,
            analysis_date=datetime.now(),
            analyst=analyst,
            bigdata_result=first_result.bigdata_result,
            image_result=image_result,
            graphrag_result=graphrag_result,
            past_case_result=past_case_result,
            defect_classification=first_result.defect_classification,
            root_cause=root_cause,
            responsible_dept=first_result.responsible_dept,
            confidence_score=0.85,
            countermeasures=countermeasures,
            prevention_measures=prevention_measures,
            related_lots=first_result.related_lots,
            related_equipment=first_result.related_equipment,
            similar_cases=first_result.similar_cases
        )

        self.results[analysis_id] = result
        self._save_results()

        logger.info(f"2차 분석 완료: {analysis_id}")
        return result

    def _analyze_bigdata(self, defect_type: str, lot_id: str, product_model: str) -> Dict[str, Any]:
        """빅데이터 기반 분석"""
        result = {
            "analysis_method": "빅데이터 상관분석",
            "defect_type": defect_type,
            "lot_id": lot_id,
            "product_model": product_model,
            "related_lots": [],
            "related_equipment": [],
            "defect_rate_trend": [],
            "confidence": 0.0
        }

        # 데이터마트에서 결함 요약 정보 로드
        defect_summary_file = os.path.join(self.datamart_dir, "defect_summary.json")
        if os.path.exists(defect_summary_file):
            with open(defect_summary_file, 'r', encoding='utf-8') as f:
                defect_summary = json.load(f)

            # 결함 유형별 통계
            defect_mapping = {
                "DEAD_PIXEL": "SPOT",
                "BRIGHT_SPOT": "SPOT",
                "LINE_DEFECT": "LINE",
                "MURA": "MURA",
                "SCRATCH": "SCRATCH",
                "TOUCH_FAIL": "PATTERN_DEFECT"
            }

            mapped_type = defect_mapping.get(defect_type, defect_type)
            if mapped_type in defect_summary:
                stats = defect_summary[mapped_type]
                result["defect_statistics"] = {
                    "total_count": stats.get("count", 0),
                    "severity_distribution": stats.get("severities", {})
                }
                result["confidence"] = 0.78

        # 설비별 불량률 정보 로드
        equipment_file = os.path.join(self.datamart_dir, "equipment_defect_rate.json")
        if os.path.exists(equipment_file):
            with open(equipment_file, 'r', encoding='utf-8') as f:
                equipment_data = json.load(f)

            # 불량률 높은 설비 추출
            high_defect_equipment = [
                eq["equipment_id"] for eq in equipment_data[:5]
            ]
            result["related_equipment"] = high_defect_equipment
            result["equipment_analysis"] = {
                "high_risk_equipment": equipment_data[:3] if equipment_data else []
            }

        # 연관 Lot 추정 (목업)
        result["related_lots"] = [
            f"{lot_id}_REL1",
            f"{lot_id}_REL2",
            f"{lot_id}_REL3"
        ]

        # 트렌드 분석 (목업)
        result["defect_rate_trend"] = [
            {"period": "2024-W48", "rate": 2.1},
            {"period": "2024-W49", "rate": 2.3},
            {"period": "2024-W50", "rate": 2.8},
            {"period": "2024-W51", "rate": 3.2},
            {"period": "2024-W52", "rate": 2.9}
        ]

        return result

    def _estimate_responsible_dept(self, defect_type: str, bigdata_result: Dict) -> str:
        """귀책 부서 추정"""
        # 결함 유형별 귀책 부서 매핑
        dept_mapping = {
            "DEAD_PIXEL": "TFT공정",
            "BRIGHT_SPOT": "TFT공정",
            "LINE_DEFECT": "TFT공정",
            "MURA": "CF공정",
            "SCRATCH": "CELL공정",
            "TOUCH_FAIL": "모듈공정",
            "PATTERN_DEFECT": "TFT공정",
            "SPOT": "TFT공정",
            "STAIN": "CF공정",
            "CRACK": "CELL공정",
            "PARTICLE": "자재"
        }

        return dept_mapping.get(defect_type, "미정")

    def _find_similar_cases(self, defect_type: str, product_model: str) -> List[str]:
        """유사 사례 검색"""
        # 목업 데이터 - 실제로는 GraphRAG에서 검색
        similar_cases = [
            f"CASE_{defect_type}_2024001",
            f"CASE_{defect_type}_2024002",
            f"CASE_{defect_type}_2023045"
        ]
        return similar_cases

    def _analyze_defect_image(self, image_path: str) -> Dict[str, Any]:
        """결함 이미지 분석 (Cosmos VLM 연동)"""
        # 실제로는 Cosmos VLM API 호출
        result = {
            "analysis_method": "Cosmos VLM 이미지 분석",
            "image_path": image_path,
            "detected_defects": [
                {
                    "type": "point_defect",
                    "location": {"x": 512, "y": 384},
                    "size_mm": 0.3,
                    "confidence": 0.92
                }
            ],
            "defect_characteristics": {
                "shape": "circular",
                "color": "dark",
                "pattern": "isolated",
                "estimated_cause": "TFT 공정 중 particle 유입 추정"
            },
            "severity_assessment": "MAJOR",
            "vlm_confidence": 0.89
        }
        return result

    def _analyze_with_graphrag(self, defect_type: str, bigdata_result: Dict) -> Dict[str, Any]:
        """GraphRAG 기반 원인/대책 분석"""
        # 실제로는 GraphRAG API 호출
        result = {
            "analysis_method": "GraphRAG 지식그래프 분석",
            "defect_type": defect_type,
            "cause_analysis": {
                "primary_cause": "TFT Array 공정 중 PR 코팅 불균일",
                "secondary_causes": [
                    "Spin coater 회전속도 변동",
                    "Clean room 습도 관리 미흡",
                    "PR 소재 lot 변경"
                ],
                "evidence": [
                    "동일 설비 사용 Lot에서 유사 불량 다수 발생",
                    "특정 시간대 집중 발생 패턴"
                ]
            },
            "effect_analysis": {
                "display_impact": "화면 특정 영역 시인성 저하",
                "customer_impact": "사용자 불만 야기",
                "quality_grade": "B-Grade 판정"
            },
            "recommended_actions": [
                "Spin coater 정비 및 회전속도 재설정",
                "Clean room 습도 모니터링 강화",
                "PR 소재 입고검사 기준 강화"
            ],
            "related_nodes": [
                {"id": "n001", "type": "equipment", "name": "SPIN_COATER_01"},
                {"id": "n002", "type": "material", "name": "PR_MATERIAL_A"},
                {"id": "n003", "type": "process", "name": "TFT_ARRAY"}
            ],
            "confidence": 0.82
        }
        return result

    def _analyze_past_cases(self, defect_type: str, similar_cases: List[str]) -> Dict[str, Any]:
        """과거 사례 분석"""
        # 과거 사례 문서 참조 (목업)
        result = {
            "analysis_method": "과거 사례 분석",
            "total_similar_cases": len(similar_cases),
            "cases_analyzed": [],
            "common_patterns": [],
            "effective_measures": []
        }

        # 샘플 과거 사례
        sample_cases = [
            {
                "case_id": similar_cases[0] if similar_cases else "CASE_001",
                "occurrence_date": "2024-09-15",
                "customer": "APPLE",
                "defect_type": defect_type,
                "root_cause": "설비 PM 주기 초과로 인한 particle 증가",
                "countermeasure": "PM 주기 단축 (4주 → 2주)",
                "result": "불량률 50% 감소",
                "similarity_score": 0.89
            },
            {
                "case_id": similar_cases[1] if len(similar_cases) > 1 else "CASE_002",
                "occurrence_date": "2024-07-22",
                "customer": "SAMSUNG",
                "defect_type": defect_type,
                "root_cause": "자재 lot 변경 시 spec 미확인",
                "countermeasure": "자재 변경 시 SPC 강화",
                "result": "재발 방지 완료",
                "similarity_score": 0.75
            }
        ]

        result["cases_analyzed"] = sample_cases
        result["common_patterns"] = [
            "설비 관리 미흡",
            "자재 변경 관리 필요",
            "공정 파라미터 모니터링 강화 필요"
        ]
        result["effective_measures"] = [
            "PM 주기 관리 강화",
            "자재 변경 점 관리 프로세스 수립",
            "SPC 모니터링 항목 추가"
        ]

        return result

    def _determine_root_cause(self,
                              bigdata_result: Dict,
                              image_result: Dict,
                              graphrag_result: Dict,
                              past_case_result: Dict) -> str:
        """근본 원인 결정"""
        # 각 분석 결과 종합
        causes = []

        if graphrag_result.get("cause_analysis", {}).get("primary_cause"):
            causes.append(graphrag_result["cause_analysis"]["primary_cause"])

        if image_result.get("defect_characteristics", {}).get("estimated_cause"):
            causes.append(image_result["defect_characteristics"]["estimated_cause"])

        if past_case_result.get("cases_analyzed"):
            for case in past_case_result["cases_analyzed"][:2]:
                if case.get("root_cause"):
                    causes.append(case["root_cause"])

        # 가장 높은 신뢰도의 원인 선택 (실제로는 더 정교한 로직 필요)
        if causes:
            return causes[0]
        return "원인 분석 중"

    def _generate_countermeasures(self,
                                   root_cause: str,
                                   graphrag_result: Dict,
                                   past_case_result: Dict) -> List[str]:
        """대책 생성"""
        countermeasures = []

        # GraphRAG 권장 대책
        if graphrag_result.get("recommended_actions"):
            countermeasures.extend(graphrag_result["recommended_actions"])

        # 과거 사례 효과적 대책
        if past_case_result.get("effective_measures"):
            for measure in past_case_result["effective_measures"]:
                if measure not in countermeasures:
                    countermeasures.append(measure)

        return countermeasures[:5]  # 상위 5개

    def _generate_prevention_measures(self,
                                       root_cause: str,
                                       graphrag_result: Dict) -> List[str]:
        """재발 방지 대책 생성"""
        prevention = [
            "해당 설비 집중 모니터링 체계 구축",
            "유사 공정 라인 전수 점검 실시",
            "작업 표준서 개정 및 작업자 재교육",
            "품질 관리 기준 강화 (Spec 조정)",
            "협력업체 품질 관리 강화"
        ]

        return prevention

    def get_analysis_result(self, analysis_id: str) -> Optional[AnalysisResult]:
        """분석 결과 조회"""
        return self.results.get(analysis_id)

    def get_results_by_complaint(self, complaint_id: str) -> List[AnalysisResult]:
        """불만 ID로 분석 결과 조회"""
        return [r for r in self.results.values() if r.complaint_id == complaint_id]
