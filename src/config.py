import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
CHART_DIR = OUTPUT_DIR / "charts"
MODEL_DIR = OUTPUT_DIR / "models"

DATA_URL = os.getenv("url")
if not DATA_URL:
    # .env에 url 키가 없거나 값이 비어있는 경우 다운로드 단계에서 원인을 알 수 없는 오류가 나므로 미리 안내
    raise RuntimeError(".env 파일에 'url' 값이 설정되어 있지 않습니다.")

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
