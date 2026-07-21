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

    missing_cols = df_clean.columns[df_clean.isnull().any()]
    for col in missing_cols:
        mode_series = df_clean[col].mode()
        if mode_series.empty:
            # 컬럼 전체가 결측치라 최빈값 자체가 존재하지 않는 경우
            print(f"[{col}] 전체가 결측치라 최빈값 대체를 건너뜁니다.")
            continue

        mode_value = mode_series.iloc[0]
        df_clean[col] = df_clean[col].fillna(mode_value)
        print(f"[{col}] 결측치를 최빈값({mode_value})으로 대체")

    return df_clean.reset_index(drop=True)
