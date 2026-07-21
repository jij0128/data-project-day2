import pandas as pd


def check_missing(df: pd.DataFrame) -> pd.Series:
    missing = df.isnull().sum()
    print("=== 결측치 개수 ===")
    print(missing[missing > 0] if missing.any() else "결측치가 없습니다.")
    return missing


def check_duplicates(df: pd.DataFrame) -> int:
    dup_count = int(df.duplicated().sum())
    print(f"=== 중복 행 개수: {dup_count} ===")
    return dup_count


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.drop_duplicates().copy()
    removed = len(df) - len(df_clean)
    print(f"[중복 제거] {removed}행 제거")

    before_dropna = len(df_clean)
    df_clean = df_clean.dropna()
    removed_na = before_dropna - len(df_clean)
    print(f"[결측치 제거] {removed_na}행 제거")

    return df_clean.reset_index(drop=True)
