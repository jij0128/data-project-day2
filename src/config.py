# ---------------------------------------------------------------------------
# 작성자 : 문승인, 서수영, 이명로, 최승민, 정인제, 김서현
# 작성일 : 2026-07-21
# 작성목적 : 프로젝트 공통 설정값(경로, 데이터 URL, 컬럼 목록) 정의
#
# 본파일은 Skala 교육을 위한 Sample 코드이므로 작성자에게 모든 저작권이 있습니다.
#
# 변경사항 내역(날짜, 변경목적, 변경 내용 순으로 기입)
#   - 2026-07-21, report.md 자동 생성 지원, REPORT_PATH 상수 추가
#   - 2026-07-21, .env 없이도 실행되도록 개선, DATA_URL을 필수값에서 기본값 방식으로 변경
#
# ----------------------------------------------------------------------------

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
CHART_DIR = OUTPUT_DIR / "charts"
MODEL_DIR = OUTPUT_DIR / "models"
REPORT_PATH = OUTPUT_DIR / "report.md"

# UCI Adult 데이터셋 URL은 비밀값이 아닌 공개 링크이므로 기본값으로 코드에 둔다.
# .env가 없어도(예: 다른 팀원/채점자가 새로 clone한 경우) 바로 실행되며,
# 다른 URL(미러 등)을 쓰고 싶을 때만 .env의 'url' 값으로 덮어쓰면 된다.
_DEFAULT_DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
DATA_URL = os.getenv("url", _DEFAULT_DATA_URL)

RAW_DATA_PATH = DATA_DIR / "adult.data"

# UCI Adult 데이터셋은 헤더가 없으므로 컬럼명을 직접 지정
COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education_num",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
    "native_country",
    "income",
]

NUMERIC_COLS = [
    "age",
    "fnlwgt",
    "education_num",
    "capital_gain",
    "capital_loss",
    "hours_per_week",
]
CATEGORICAL_COLS = [
    "workclass",
    "education",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]
TARGET_COL = "income"
