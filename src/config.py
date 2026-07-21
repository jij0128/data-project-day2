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
