import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd
import polars as pl

from src.config import COLUMNS, DATA_URL, RAW_DATA_PATH


def download_dataset(url: str = DATA_URL, dest_path: Path = RAW_DATA_PATH) -> Path:
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists():
        print(f"[다운로드] 캐시된 파일을 사용합니다: {dest_path}")
        return dest_path

    try:
        urllib.request.urlretrieve(url, dest_path)
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        # URL이 잘못되었거나 네트워크 연결이 안 되는 경우
        raise RuntimeError(f"데이터셋 다운로드 중 오류가 발생했습니다: {e}")

    print(f"[다운로드] 데이터셋을 저장했습니다: {dest_path}")
    return dest_path


def load_with_pandas(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(
            path,
            header=None,
            names=COLUMNS,
            na_values="?",
            skipinitialspace=True,
        )
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {path}")


def load_with_polars(path: Path) -> pl.DataFrame:
    df = pl.read_csv(
        path,
        has_header=False,
        new_columns=COLUMNS,
        null_values=["?", " ?"],
    )
    # pandas의 skipinitialspace와 달리 polars는 구분자 뒤 공백을 남기므로 문자열 컬럼만 별도로 strip
    str_cols = [name for name, dtype in df.schema.items() if dtype == pl.Utf8]
    return df.with_columns([pl.col(c).str.strip_chars() for c in str_cols])


def compare_pandas_polars(pdf: pd.DataFrame, pldf: pl.DataFrame) -> None:
    print("=== Pandas vs Polars 로딩 결과 비교 ===")
    print(f"[Pandas] shape={pdf.shape}, 메모리={pdf.memory_usage(deep=True).sum() / 1024 ** 2:.2f}MB")
    print(f"[Polars] shape={pldf.shape}, 메모리={pldf.estimated_size('mb'):.2f}MB")

    print("\n[Pandas dtypes]")
    print(pdf.dtypes)
    print("\n[Polars dtypes]")
    print(pldf.schema)
