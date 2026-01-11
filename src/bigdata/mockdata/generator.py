"""
목업 빅데이터 생성기
시연용 종합 데이터 생성
"""

import random
import json
import os
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import asdict

from .schemas import (
    DevelopmentData, EquipmentData, MaterialData,
    InspectionData, QualityData, ManufacturingData, MESData,
    TracebilityData, DevelopmentPhase, DefectSeverity, InspectionResult
)

logger = logging.getLogger(__name__)


class MockDataGenerator:
    """
    목업 빅데이터 생성기

    개발단계, 제조현장, MES 실적 데이터를 포함한 종합 시연용 데이터 생성
    """

    # 기본 상수들
    PROCESS_STEPS = ["TFT_ARRAY", "CF_ARRAY", "CELL_ASSEMBLY", "MODULE", "FINAL_TEST"]
    EQUIPMENT_TYPES = ["CVD", "ETCH", "PHOTO", "SPUTTER", "CLEAN", "INSPECTION", "TEST"]
    DEFECT_TYPES = ["SCRATCH", "PARTICLE", "PATTERN_DEFECT", "MURA", "SPOT", "LINE", "STAIN", "CRACK"]
    DEFECT_LOCATIONS = ["CENTER", "EDGE", "CORNER", "RANDOM"]
    CUSTOMERS = ["APPLE", "SAMSUNG_MOBILE", "LG_MOBILE", "XIAOMI", "HUAWEI", "GOOGLE", "META"]
    VENDORS = ["VENDOR_A", "VENDOR_B", "VENDOR_C", "VENDOR_D", "VENDOR_E"]
    LINES = ["L1", "L2", "L3", "L4", "L5"]
    SHIFTS = ["DAY", "SWING", "NIGHT"]

    def __init__(self, output_dir: str = "/tmp/mockdata"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 하위 디렉토리 생성
        self.dirs = {
            "development": os.path.join(output_dir, "development"),
            "equipment": os.path.join(output_dir, "equipment"),
            "material": os.path.join(output_dir, "material"),
            "inspection": os.path.join(output_dir, "inspection"),
            "quality": os.path.join(output_dir, "quality"),
            "manufacturing": os.path.join(output_dir, "manufacturing"),
            "mes": os.path.join(output_dir, "mes"),
            "traceability": os.path.join(output_dir, "traceability"),
            "parquet": os.path.join(output_dir, "parquet")
        }
        for d in self.dirs.values():
            os.makedirs(d, exist_ok=True)

    def generate_all(self,
                     num_lots: int = 100,
                     num_cells_per_lot: int = 50,
                     num_days: int = 30) -> Dict[str, str]:
        """
        모든 목업 데이터 생성

        Args:
            num_lots: 생성할 로트 수
            num_cells_per_lot: 로트당 셀 수
            num_days: 데이터 생성 기간 (일)

        Returns:
            생성된 파일 경로들
        """
        logger.info(f"목업 데이터 생성 시작: {num_lots} lots, {num_cells_per_lot} cells/lot, {num_days} days")

        result_paths = {}

        # 1. 개발단계 데이터 생성
        dev_data = self.generate_development_data(num_samples=200)
        result_paths["development"] = self._save_data(dev_data, "development", "development_data")

        # 2. 설비 마스터 데이터 생성
        equipment_data = self.generate_equipment_data(num_equipment=50)
        result_paths["equipment"] = self._save_data(equipment_data, "equipment", "equipment_master")

        # 3. 자재 데이터 생성
        material_data = self.generate_material_data(num_materials=100)
        result_paths["material"] = self._save_data(material_data, "material", "material_data")

        # 4. 제조현장 데이터 생성 (검사, 품질 포함)
        mfg_data = self.generate_manufacturing_data(num_lots, num_cells_per_lot, num_days)
        result_paths["manufacturing"] = self._save_data(mfg_data["manufacturing"], "manufacturing", "manufacturing_data")
        result_paths["inspection"] = self._save_data(mfg_data["inspection"], "inspection", "inspection_data")
        result_paths["quality"] = self._save_data(mfg_data["quality"], "quality", "quality_data")

        # 5. MES 실적 데이터 생성
        mes_data = self.generate_mes_data(num_lots, num_days)
        result_paths["mes"] = self._save_data(mes_data, "mes", "mes_performance")

        # 6. 이력추적 데이터 생성
        trace_data = self.generate_traceability_data(num_lots, num_cells_per_lot)
        result_paths["traceability"] = self._save_data(trace_data, "traceability", "traceability_data")

        # 7. Parquet 변환 시도
        result_paths["parquet"] = self._convert_to_parquet()

        logger.info(f"목업 데이터 생성 완료: {len(result_paths)} 카테고리")
        return result_paths

    def generate_development_data(self, num_samples: int = 200) -> List[Dict]:
        """개발단계 데이터 생성 (EVT, DVT, PVT)"""
        data = []

        phases = ["EVT", "DVT", "PVT", "MP"]
        test_types = ["OPTICAL", "ELECTRICAL", "MECHANICAL", "ENVIRONMENTAL", "RELIABILITY"]
        engineers = [f"ENG{i:03d}" for i in range(1, 21)]

        base_date = datetime.now() - timedelta(days=365)

        for i in range(num_samples):
            phase_idx = min(i // 50, 3)  # EVT->DVT->PVT->MP 순차 진행
            phase = phases[phase_idx]

            sample_date = base_date + timedelta(days=i * 2 + random.randint(0, 5))

            # 측정값 생성 - 단계별로 품질 향상
            base_quality = 0.7 + (phase_idx * 0.08)  # 단계별 품질 향상

            measurements = {
                "brightness": round(random.gauss(450 + phase_idx * 20, 15), 2),
                "contrast_ratio": round(random.gauss(1500 + phase_idx * 200, 100), 0),
                "color_gamut": round(random.gauss(base_quality * 100, 5), 2),
                "response_time": round(random.gauss(8 - phase_idx * 0.5, 0.5), 2),
                "power_consumption": round(random.gauss(5 - phase_idx * 0.3, 0.3), 2),
                "uniformity": round(random.gauss(base_quality * 100, 3), 2)
            }

            # 결과 판정
            pass_rate = base_quality + random.gauss(0, 0.1)
            result = "PASS" if pass_rate > 0.7 else "FAIL"

            dev = {
                "project_id": f"PRJT{2025 + phase_idx:04d}",
                "phase": phase,
                "sample_id": f"DEV{i:05d}",
                "test_date": sample_date.isoformat(),
                "test_type": random.choice(test_types),
                "test_result": result,
                "measurement_values": measurements,
                "engineer": random.choice(engineers),
                "notes": f"{phase} 단계 테스트 - 샘플 {i}",
                "related_issues": [f"ISSUE_{random.randint(1, 50):04d}"] if result == "FAIL" else []
            }
            data.append(dev)

        return data

    def generate_equipment_data(self, num_equipment: int = 50) -> List[Dict]:
        """설비 마스터 데이터 생성"""
        data = []

        equipment_configs = {
            "CVD": {"temp": (350, 450), "pressure": (1, 10), "gas_flow": (100, 500)},
            "ETCH": {"temp": (20, 80), "rf_power": (500, 2000), "pressure": (5, 50)},
            "PHOTO": {"exposure": (50, 200), "focus": (-1, 1), "dose": (10, 50)},
            "SPUTTER": {"power": (2000, 5000), "pressure": (2, 8), "gas_flow": (50, 200)},
            "CLEAN": {"temp": (40, 80), "flow_rate": (10, 50), "time": (30, 120)},
            "INSPECTION": {"resolution": (0.1, 1.0), "speed": (10, 100), "sensitivity": (0.5, 0.99)},
            "TEST": {"voltage": (3, 12), "current": (0.1, 2.0), "frequency": (60, 120)}
        }

        base_date = datetime.now() - timedelta(days=1000)

        eq_id = 1
        for line in self.LINES:
            for eq_type in self.EQUIPMENT_TYPES:
                num_per_type = random.randint(1, 3)
                for _ in range(num_per_type):
                    if eq_id > num_equipment:
                        break

                    install_date = base_date + timedelta(days=random.randint(0, 500))
                    last_pm = datetime.now() - timedelta(days=random.randint(1, 30))

                    config = equipment_configs.get(eq_type, {"param1": (0, 100)})
                    params = {}
                    for param, (low, high) in config.items():
                        params[param] = round(random.uniform(low, high), 2)

                    status = random.choices(
                        ["RUNNING", "IDLE", "PM", "DOWN"],
                        weights=[0.7, 0.15, 0.1, 0.05]
                    )[0]

                    eq = {
                        "equipment_id": f"EQ{eq_id:04d}",
                        "equipment_name": f"{eq_type}_{line}_{eq_id:02d}",
                        "equipment_type": eq_type,
                        "line_id": line,
                        "vendor": random.choice(self.VENDORS),
                        "install_date": install_date.isoformat(),
                        "last_pm_date": last_pm.isoformat(),
                        "status": status,
                        "utilization": round(random.uniform(0.6, 0.95) if status == "RUNNING" else 0, 2),
                        "parameters": params
                    }
                    data.append(eq)
                    eq_id += 1

        return data

    def generate_material_data(self, num_materials: int = 100) -> List[Dict]:
        """자재 데이터 생성"""
        data = []

        material_types = {
            "GLASS": {"thickness": (0.3, 0.7), "size": (1500, 2500), "flatness": (0.01, 0.1)},
            "RESIST": {"viscosity": (5, 20), "solid_content": (20, 40), "ph": (6.5, 7.5)},
            "ETCHANT": {"concentration": (10, 50), "temp": (20, 40), "purity": (99.0, 99.99)},
            "TARGET": {"purity": (99.9, 99.999), "density": (5, 15), "thickness": (5, 15)},
            "GAS": {"purity": (99.99, 99.9999), "pressure": (50, 200), "flow_rate": (100, 1000)},
            "POLARIZER": {"transmittance": (40, 50), "thickness": (100, 200), "haze": (0.01, 0.1)},
            "ADHESIVE": {"viscosity": (10, 100), "cure_time": (5, 30), "strength": (10, 50)}
        }

        base_date = datetime.now() - timedelta(days=60)

        for i in range(num_materials):
            mat_type = random.choice(list(material_types.keys()))
            specs = material_types[mat_type]

            receive_date = base_date + timedelta(days=random.randint(0, 50))
            expire_date = receive_date + timedelta(days=random.randint(90, 365))

            spec_values = {}
            for spec, (low, high) in specs.items():
                spec_values[spec] = round(random.uniform(low, high), 4)

            mat = {
                "material_id": f"MAT{i:05d}",
                "material_name": f"{mat_type}_{random.choice(['A', 'B', 'C', 'D'])}_{random.randint(100, 999)}",
                "material_type": mat_type,
                "lot_no": f"MLOT{datetime.now().strftime('%Y%m%d')}{i:04d}",
                "vendor": random.choice(self.VENDORS),
                "receive_date": receive_date.isoformat(),
                "expire_date": expire_date.isoformat(),
                "quantity": round(random.uniform(10, 1000), 2),
                "unit": random.choice(["kg", "L", "ea", "m2", "pcs"]),
                "quality_grade": random.choice(["A", "A", "A", "B", "B", "C"]),
                "storage_location": f"WH{random.randint(1, 5):02d}-R{random.randint(1, 20):02d}-S{random.randint(1, 10):02d}",
                "specifications": spec_values
            }
            data.append(mat)

        return data

    def generate_manufacturing_data(self, num_lots: int, num_cells: int, num_days: int) -> Dict[str, List[Dict]]:
        """제조현장 데이터 생성 (검사, 양/불량 포함)"""
        manufacturing_data = []
        inspection_data = []
        quality_data = []

        base_date = datetime.now() - timedelta(days=num_days)

        for lot_idx in range(num_lots):
            lot_id = f"LOT{(base_date + timedelta(days=lot_idx % num_days)).strftime('%Y%m%d')}{lot_idx:04d}"
            customer = random.choice(self.CUSTOMERS)
            line = random.choice(self.LINES)

            # 로트별 기본 품질 특성 (로트마다 약간 다름)
            lot_quality_base = random.gauss(0.92, 0.03)

            for cell_idx in range(num_cells):
                cell_id = f"{lot_id}_C{cell_idx:04d}"
                process_time = base_date + timedelta(
                    days=lot_idx % num_days,
                    hours=cell_idx * 0.5 + random.random() * 2
                )

                for step_idx, step in enumerate(self.PROCESS_STEPS):
                    step_time = process_time + timedelta(hours=step_idx * 4)
                    eq_id = f"EQ{random.randint(1, 50):04d}"

                    # 검사 결과 생성
                    quality_factor = lot_quality_base + random.gauss(0, 0.02)
                    is_pass = random.random() < quality_factor

                    defect_codes = []
                    defect_type = ""
                    defect_severity = ""

                    if not is_pass:
                        num_defects = random.randint(1, 3)
                        defect_codes = random.sample(self.DEFECT_TYPES, num_defects)
                        defect_type = defect_codes[0]
                        defect_severity = random.choice(["CRITICAL", "MAJOR", "MINOR", "COSMETIC"])

                    # 검사 데이터
                    insp = {
                        "inspection_id": f"INS{lot_idx:04d}{cell_idx:04d}{step_idx:02d}",
                        "lot_id": lot_id,
                        "cell_id": cell_id,
                        "inspection_type": random.choice(["AOI", "VI", "EL", "PHOTO"]),
                        "inspection_step": step,
                        "inspector_id": f"OP{random.randint(1, 50):03d}",
                        "inspection_time": step_time.isoformat(),
                        "result": "PASS" if is_pass else "FAIL",
                        "defect_codes": defect_codes,
                        "measurements": {
                            "brightness": round(random.gauss(450, 20), 2),
                            "uniformity": round(random.gauss(95, 3), 2),
                            "contrast": round(random.gauss(1500, 100), 0)
                        },
                        "images": [f"/images/{cell_id}_{step}_{i}.png" for i in range(random.randint(1, 4))]
                    }
                    inspection_data.append(insp)

                    # 품질 데이터
                    qual = {
                        "lot_id": lot_id,
                        "cell_id": cell_id,
                        "process_step": step,
                        "inspection_result": "PASS" if is_pass else "FAIL",
                        "defect_type": defect_type,
                        "defect_location": random.choice(self.DEFECT_LOCATIONS) if defect_type else "",
                        "defect_size": round(random.uniform(0.1, 5.0), 2) if defect_type else 0,
                        "severity": defect_severity,
                        "root_cause": random.choice(["EQUIPMENT", "MATERIAL", "PROCESS", "OPERATOR", "UNKNOWN"]) if defect_type else "",
                        "corrective_action": "REWORK" if defect_severity in ["MINOR", "COSMETIC"] else "SCRAP" if defect_type else "",
                        "timestamp": step_time.isoformat(),
                        "equipment_id": eq_id,
                        "operator_id": f"OP{random.randint(1, 50):03d}"
                    }
                    quality_data.append(qual)

                    # 제조현장 통합 데이터 (마지막 공정만 대표로)
                    if step_idx == len(self.PROCESS_STEPS) - 1:
                        mfg = {
                            "lot_id": lot_id,
                            "wafer_id": f"{lot_id}_W{cell_idx // 10:02d}",
                            "cell_id": cell_id,
                            "process_step": step,
                            "equipment_id": eq_id,
                            "material_lots": [f"MLOT{random.randint(1, 100):05d}" for _ in range(random.randint(2, 5))],
                            "process_time": step_time.isoformat(),
                            "cycle_time": round(random.gauss(120, 10), 2),
                            "yield_rate": round(quality_factor * 100, 2),
                            "final_result": "PASS" if is_pass else "FAIL",
                            "customer": customer,
                            "line_id": line
                        }
                        manufacturing_data.append(mfg)

        return {
            "manufacturing": manufacturing_data,
            "inspection": inspection_data,
            "quality": quality_data
        }

    def generate_mes_data(self, num_lots: int, num_days: int) -> List[Dict]:
        """MES 실적 데이터 생성"""
        data = []

        base_date = datetime.now() - timedelta(days=num_days)
        products = [
            ("PROD_A", "OLED 6.7\" FHD+", 0.94),
            ("PROD_B", "OLED 6.1\" FHD", 0.95),
            ("PROD_C", "LTPO 6.8\" QHD+", 0.92),
            ("PROD_D", "LCD 10.9\" 2K", 0.96),
            ("PROD_E", "OLED 7.6\" Foldable", 0.88)
        ]

        for lot_idx in range(num_lots):
            lot_date = base_date + timedelta(days=lot_idx % num_days)
            lot_id = f"LOT{lot_date.strftime('%Y%m%d')}{lot_idx:04d}"

            product = random.choice(products)
            product_id, product_name, base_yield = product

            plan_qty = random.choice([100, 200, 500, 1000])
            actual_yield = base_yield + random.gauss(0, 0.02)
            actual_qty = int(plan_qty * random.uniform(0.98, 1.02))
            good_qty = int(actual_qty * actual_yield)
            ng_qty = actual_qty - good_qty

            start_time = lot_date + timedelta(hours=random.randint(6, 10))
            end_time = start_time + timedelta(hours=random.randint(8, 24))

            mes = {
                "work_order_id": f"WO{lot_date.strftime('%Y%m%d')}{lot_idx:04d}",
                "lot_id": lot_id,
                "product_id": product_id,
                "product_name": product_name,
                "customer": random.choice(self.CUSTOMERS),
                "plan_qty": plan_qty,
                "actual_qty": actual_qty,
                "good_qty": good_qty,
                "ng_qty": ng_qty,
                "yield_rate": round(actual_yield * 100, 2),
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "line_id": random.choice(self.LINES),
                "shift": random.choice(self.SHIFTS),
                "operator_id": f"OP{random.randint(1, 50):03d}",
                "process_route": self.PROCESS_STEPS,
                "status": "COMPLETED" if lot_idx < num_lots - 5 else random.choice(["IN_PROGRESS", "WAITING", "COMPLETED"]),
                "remarks": f"정상 생산 완료" if actual_yield > 0.90 else f"수율 저하 - {random.choice(self.DEFECT_TYPES)} 발생"
            }
            data.append(mes)

        return data

    def generate_traceability_data(self, num_lots: int, num_cells: int) -> List[Dict]:
        """이력추적 데이터 생성"""
        data = []

        base_date = datetime.now() - timedelta(days=30)

        # 주요 셀에 대한 이력 추적 데이터
        for lot_idx in range(min(num_lots, 20)):  # 상위 20개 로트만
            lot_id = f"LOT{base_date.strftime('%Y%m%d')}{lot_idx:04d}"

            for cell_idx in range(min(num_cells, 10)):  # 로트당 10개 셀만
                cell_id = f"{lot_id}_C{cell_idx:04d}"

                for step_idx, step in enumerate(self.PROCESS_STEPS):
                    timestamp = base_date + timedelta(
                        days=lot_idx,
                        hours=step_idx * 4 + random.random() * 2
                    )

                    # 일부 셀에 결함 연결
                    has_defect = random.random() < 0.1

                    trace = {
                        "trace_id": f"TRC{lot_idx:04d}{cell_idx:04d}{step_idx:02d}",
                        "cell_id": cell_id,
                        "lot_id": lot_id,
                        "timestamp": timestamp.isoformat(),
                        "event_type": step,
                        "event_detail": f"{step} 공정 완료",
                        "equipment_id": f"EQ{random.randint(1, 50):04d}",
                        "material_lots": [f"MLOT{random.randint(1, 100):05d}" for _ in range(random.randint(1, 3))],
                        "parameters": {
                            "temp": round(random.gauss(350, 20), 2),
                            "pressure": round(random.gauss(5, 1), 2),
                            "time": round(random.gauss(60, 10), 2)
                        },
                        "quality_result": "FAIL" if has_defect else "PASS",
                        "linked_defects": [f"DEF{random.randint(1, 1000):05d}"] if has_defect else []
                    }
                    data.append(trace)

        return data

    def _save_data(self, data: List[Dict], category: str, filename: str) -> str:
        """데이터를 JSON 파일로 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(self.dirs[category], f"{filename}_{timestamp}.json")

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"데이터 저장 완료: {filepath} ({len(data)} records)")
        return filepath

    def _convert_to_parquet(self) -> str:
        """JSON 데이터를 Parquet로 변환 시도"""
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq

            parquet_files = []

            for category, dir_path in self.dirs.items():
                if category == "parquet":
                    continue

                for json_file in os.listdir(dir_path):
                    if json_file.endswith(".json"):
                        json_path = os.path.join(dir_path, json_file)
                        parquet_path = os.path.join(
                            self.dirs["parquet"],
                            json_file.replace(".json", ".parquet")
                        )

                        with open(json_path, "r", encoding="utf-8") as f:
                            data = json.load(f)

                        if data:
                            # 중첩 구조 평탄화
                            flat_data = []
                            for record in data:
                                flat_record = {}
                                for key, value in record.items():
                                    if isinstance(value, (dict, list)):
                                        flat_record[key] = json.dumps(value, ensure_ascii=False)
                                    else:
                                        flat_record[key] = value
                                flat_data.append(flat_record)

                            table = pa.Table.from_pylist(flat_data)
                            pq.write_table(
                                table,
                                parquet_path,
                                compression="snappy"
                            )
                            parquet_files.append(parquet_path)

            logger.info(f"Parquet 변환 완료: {len(parquet_files)} files")
            return self.dirs["parquet"]

        except ImportError:
            logger.warning("pyarrow가 설치되지 않아 Parquet 변환을 건너뜁니다.")
            return ""

    def get_summary(self) -> Dict[str, Any]:
        """생성된 데이터 요약 정보"""
        summary = {}

        for category, dir_path in self.dirs.items():
            files = []
            total_records = 0

            for filename in os.listdir(dir_path):
                filepath = os.path.join(dir_path, filename)

                if filename.endswith(".json"):
                    with open(filepath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        total_records += len(data)

                files.append({
                    "filename": filename,
                    "size": os.path.getsize(filepath)
                })

            summary[category] = {
                "directory": dir_path,
                "files": files,
                "total_records": total_records
            }

        return summary
