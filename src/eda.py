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
