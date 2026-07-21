# ---------------------------------------------------------------------------
# 작성자 : 문승인, 서수영, 이명로, 최승민, 정인제, 김서현
# 작성일 : 2026-07-21
# 작성목적 : 정제 전 원본 데이터에 대한 기본 EDA(shape·info·head·기술통계·범주형 분포) 수행
#
# 본파일은 Skala 교육을 위한 Sample 코드이므로 작성자에게 모든 저작권이 있습니다.
#
# 변경사항 내역(날짜, 변경목적, 변경 내용 순으로 기입)
#   - 2026-07-21, 완성도(주석) 보완, run_eda에 docstring 추가
#
# ----------------------------------------------------------------------------

import pandas as pd


def run_eda(df: pd.DataFrame) -> None:
    """정제 전 원본 데이터의 shape·info·head·기술통계·범주형 값 분포를 훑어보는 기본 EDA."""
    print("=== shape ===")
    print(df.shape)

    print("\n=== info ===")
    print(df.info())

    print("\n=== head ===")
    print(df.head(3))

    print("\n=== 수치형 컬럼 기술통계 ===")
    print(df.describe())

    print("\n=== 범주형 컬럼 값 개수 (상위 5개) ===")
    for col in df.select_dtypes(include="object").columns:
        print(f"\n[{col}]")
        print(df[col].value_counts(dropna=False).head(5))
