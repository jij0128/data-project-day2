import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd
import polars as pl

from src.config import COLUMNS, DATA_URL, NUMERIC_COLS, RAW_DATA_PATH


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
    # 원본 CSV가 ", "(콤마+공백)로 구분되어 있는데, pandas의 skipinitialspace와 달리 polars는
    # 구분자 뒤 공백을 자동으로 제거하지 않는다. 그래서 " 77516"처럼 앞에 공백이 붙은 값은
    # 숫자로 보여도 polars가 전부 문자열(Utf8)로 인식해버린다 (맨 앞 컬럼인 age만 공백이 없어 정상 인식됨).
    str_cols = [name for name, dtype in df.schema.items() if dtype == pl.Utf8]
    df = df.with_columns([pl.col(c).str.strip_chars() for c in str_cols])

    # 공백 제거 후에도 문자열로 남아있는 수치형 컬럼(NUMERIC_COLS)은 원래 타입(정수)으로 되돌린다.
    # 이걸 안 해주면 fnlwgt/education_num/capital_gain/capital_loss/hours_per_week가
    # pandas에서는 int64인데 polars에서는 계속 문자열로 남아 두 결과가 어긋난다.
    numeric_str_cols = [c for c in NUMERIC_COLS if c in str_cols]
    df = df.with_columns([pl.col(c).cast(pl.Int64) for c in numeric_str_cols])

    # adult.data 파일 맨 끝의 빈 줄이 모든 컬럼이 null인 행 하나로 읽혀 pandas보다 1행 많아지므로 제거한다.
    return df.filter(pl.col(COLUMNS[0]).is_not_null())


def compare_pandas_polars(pdf: pd.DataFrame, pldf: pl.DataFrame) -> None:
    print("=== Pandas vs Polars 로딩 결과 비교 ===")
    print(
        f"[Pandas] shape={pdf.shape}, 메모리={pdf.memory_usage(deep=True).sum() / 1024**2:.2f}MB"
    )
    print(f"[Polars] shape={pldf.shape}, 메모리={pldf.estimated_size('mb'):.2f}MB")

    print("\n[Pandas dtypes]")
    print(pdf.dtypes)
    print("\n[Polars dtypes]")
    print(pldf.schema)

    # 두 도구가 같은 원본을 읽고도 행 수가 다르면 로더 설정(공백 처리, 빈 줄 등)이 어긋났다는 뜻이므로 여기서 바로 잡아낸다.
    assert pdf.shape[0] == pldf.shape[0], (
        f"Pandas/Polars 로딩 결과의 행 수가 다릅니다: {pdf.shape[0]} vs {pldf.shape[0]}"
    )
    print("\n[검증] Pandas/Polars 로딩 결과 행 수 일치 확인 완료")
