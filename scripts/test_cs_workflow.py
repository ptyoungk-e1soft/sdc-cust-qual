#!/usr/bin/env python3
"""
CS 워크플로우 테스트 스크립트
전체 불량 분석 프로세스 시연
"""

import sys
import os
import json
from datetime import datetime

# 프로젝트 루트 경로 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.bigdata.cs_workflow.cs_complaint import CSComplaint, CSComplaintManager, ComplaintStatus, ResponsibleDept
from src.bigdata.cs_workflow.quality_analysis import QualityAnalyzer, AnalysisResult
from src.bigdata.cs_workflow.report_generator import ReportGenerator


def print_section(title: str):
    """섹션 구분선 출력"""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


def print_json(data: dict, indent: int = 2):
    """JSON 포맷 출력"""
    print(json.dumps(data, ensure_ascii=False, indent=indent, default=str))


def main():
    print_section("CS 품질 불량 분석 워크플로우 시연")
    print(f"시연 시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 1. 초기화
    print_section("1. 시스템 초기화")

    complaint_manager = CSComplaintManager(data_dir="/tmp/cs_complaints")
    quality_analyzer = QualityAnalyzer(data_dir="/tmp/quality_analysis")
    report_generator = ReportGenerator(output_dir="/tmp/cs_reports")

    print("✓ CSComplaintManager 초기화 완료")
    print("✓ QualityAnalyzer 초기화 완료")
    print("✓ ReportGenerator 초기화 완료")

    # 2. 과거 사례 문서 생성
    print_section("2. 과거 사례 문서 생성")

    past_case_files = report_generator.generate_sample_past_cases(count=5)
    print(f"✓ 과거 사례 문서 {len(past_case_files)}건 생성 완료")
    for f in past_case_files:
        print(f"  - {os.path.basename(f)}")

    # 3. CS 불만 접수
    print_section("3. CS 불만 접수 (시나리오: APPLE Dead Pixel)")

    complaint = complaint_manager.create_complaint(
        customer="APPLE",
        product_model="OLED_67_FHD",
        lot_id="LOT20241203001",
        cell_id="CELL12345",
        defect_type="DEAD_PIXEL",
        defect_description="화면 중앙부에 검은색 점 발견. 크기 약 0.3mm, 백색 배경에서 육안 확인 가능",
        defect_image_path="/data/defect_images/dead_pixel_sample.png",
        severity="HIGH"
    )

    print(f"✓ 불만 접수 완료")
    print(f"  - 접수번호: {complaint.complaint_id}")
    print(f"  - 고객사: {complaint.customer}")
    print(f"  - 제품: {complaint.product_model}")
    print(f"  - 결함유형: {complaint.defect_type}")
    print(f"  - 심각도: {complaint.severity}")
    print(f"  - 상태: {complaint.status}")

    # 4. 1차 기본 분석 (품질부서)
    print_section("4. 1차 기본 분석 (품질부서)")

    first_analysis = quality_analyzer.perform_first_analysis(
        complaint_id=complaint.complaint_id,
        defect_type=complaint.defect_type,
        lot_id=complaint.lot_id,
        cell_id=complaint.cell_id,
        product_model=complaint.product_model,
        analyst="QA_Analyst_01"
    )

    # 불만 데이터 업데이트
    complaint_manager.update_first_analysis(
        complaint_id=complaint.complaint_id,
        result=first_analysis.to_dict(),
        analyst="QA_Analyst_01"
    )

    print(f"✓ 1차 분석 완료")
    print(f"  - 분석 ID: {first_analysis.analysis_id}")
    print(f"  - 귀책 부서 추정: {first_analysis.responsible_dept}")
    print(f"  - 신뢰도: {first_analysis.confidence_score*100:.1f}%")
    print(f"  - 연관 설비: {first_analysis.related_equipment}")
    print(f"  - 유사 사례: {first_analysis.similar_cases}")

    print("\n■ 빅데이터 분석 결과:")
    print_json(first_analysis.bigdata_result)

    # 5. 귀책 부서 배정
    print_section("5. 귀책 부서 배정")

    # 귀책 부서 매핑
    dept_mapping = {
        "TFT공정": ResponsibleDept.TFT,
        "CF공정": ResponsibleDept.CF,
        "CELL공정": ResponsibleDept.CELL,
        "모듈공정": ResponsibleDept.MODULE,
        "자재": ResponsibleDept.MATERIAL,
        "설비": ResponsibleDept.EQUIPMENT,
        "설계": ResponsibleDept.DESIGN
    }

    responsible_dept = dept_mapping.get(first_analysis.responsible_dept, ResponsibleDept.UNKNOWN)

    complaint_manager.assign_responsible_dept(
        complaint_id=complaint.complaint_id,
        dept=responsible_dept,
        analyst="Dept_Analyst_03"
    )

    print(f"✓ 귀책 부서 배정 완료")
    print(f"  - 귀책 부서: {responsible_dept.value}")
    print(f"  - 담당 분석자: Dept_Analyst_03")

    # 6. 2차 상세 분석 (귀책부서)
    print_section("6. 2차 상세 분석 (귀책부서)")

    second_analysis = quality_analyzer.perform_second_analysis(
        complaint_id=complaint.complaint_id,
        first_analysis_id=first_analysis.analysis_id,
        defect_image_path=complaint.defect_image_path,
        analyst="Dept_Analyst_03"
    )

    # 불만 데이터 업데이트
    complaint_manager.update_second_analysis(
        complaint_id=complaint.complaint_id,
        result=second_analysis.to_dict(),
        root_cause=second_analysis.root_cause,
        countermeasure=", ".join(second_analysis.countermeasures[:3])
    )

    print(f"✓ 2차 분석 완료")
    print(f"  - 분석 ID: {second_analysis.analysis_id}")
    print(f"  - 근본 원인: {second_analysis.root_cause}")
    print(f"  - 신뢰도: {second_analysis.confidence_score*100:.1f}%")

    print("\n■ 이미지 분석 결과:")
    print_json(second_analysis.image_result)

    print("\n■ GraphRAG 분석 결과:")
    print_json(second_analysis.graphrag_result)

    print("\n■ 과거 사례 분석 결과:")
    print_json(second_analysis.past_case_result)

    print("\n■ 대책:")
    for i, measure in enumerate(second_analysis.countermeasures, 1):
        print(f"  {i}. {measure}")

    print("\n■ 재발 방지 대책:")
    for i, measure in enumerate(second_analysis.prevention_measures, 1):
        print(f"  {i}. {measure}")

    # 7. 최종 보고서 생성
    print_section("7. 최종 보고서 생성")

    # 최신 불만 데이터 가져오기
    updated_complaint = complaint_manager.get_complaint(complaint.complaint_id)

    report_path = report_generator.generate_final_report(
        complaint_data=updated_complaint.to_dict(),
        first_analysis=first_analysis.to_dict(),
        second_analysis=second_analysis.to_dict(),
        include_images=True
    )

    print(f"✓ 최종 보고서 생성 완료")
    print(f"  - 파일 경로: {report_path}")

    report_summary = report_generator.get_report_summary(report_path)
    print(f"  - 파일 형식: {report_summary.get('format', 'Unknown')}")
    print(f"  - 파일 크기: {report_summary.get('size_kb', 0):.2f} KB")

    # 8. 불만 처리 완료
    print_section("8. 불만 처리 완료")

    complaint_manager.complete_complaint(complaint.complaint_id)
    final_complaint = complaint_manager.get_complaint(complaint.complaint_id)

    print(f"✓ 불만 처리 완료")
    print(f"  - 최종 상태: {final_complaint.status}")
    print(f"  - 완료 일시: {final_complaint.completion_date}")

    # 9. 통계 요약
    print_section("9. 시연 결과 요약")

    all_complaints = complaint_manager.get_all_complaints()
    print(f"  - 총 불만 건수: {len(all_complaints)}")

    status_counts = {}
    for c in all_complaints:
        status_counts[c.status] = status_counts.get(c.status, 0) + 1

    print("  - 상태별 현황:")
    for status, count in status_counts.items():
        print(f"    • {status}: {count}건")

    past_cases = report_generator.load_past_cases()
    print(f"  - 과거 사례 문서: {len(past_cases)}건")

    print("\n" + "=" * 60)
    print("  시연 완료!")
    print("=" * 60)
    print(f"\n생성된 파일:")
    print(f"  • 최종 보고서: {report_path}")
    print(f"  • 과거 사례: {report_generator.past_cases_dir}")
    print(f"  • 불만 데이터: {complaint_manager.data_dir}")
    print(f"  • 분석 결과: {quality_analyzer.data_dir}")

    return report_path


if __name__ == "__main__":
    try:
        report_path = main()
        print(f"\n✓ 테스트 성공! 보고서: {report_path}")
    except Exception as e:
        print(f"\n✗ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
