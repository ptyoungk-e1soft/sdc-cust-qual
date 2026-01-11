"""데이터 상수 정의"""

# 결함 유형 정의
DEFECT_TYPES = {
    "dead_pixel": "데드 픽셀",
    "bright_spot": "휘점 결함",
    "line_defect": "라인 결함",
    "mura": "무라 (얼룩)",
    "scratch": "스크래치",
    "particle": "이물질",
    "custom": "기타 결함",
}

SEVERITY_LEVELS = ["low", "medium", "high", "critical"]
