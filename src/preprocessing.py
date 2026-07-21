import pandas as pd


def check_missing(df: pd.DataFrame) -> pd.Series:
    """컬럼별 결측치 개수를 출력하고 반환한다 (제거 전 현황 확인용)."""
    missing = df.isnull().sum()
    print("=== 결측치 개수 ===")
    print(missing[missing > 0] if missing.any() else "결측치가 없습니다.")
    return missing


def check_duplicates(df: pd.DataFrame) -> int:
    """완전히 동일한 중복 행 개수를 출력하고 반환한다 (제거 전 현황 확인용)."""
    dup_count = int(df.duplicated().sum())
    print(f"=== 중복 행 개수: {dup_count} ===")
    return dup_count


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """중복 행과 결측치가 있는 행을 제거한다 (대체/imputation 대신 제거 방식으로 팀 논의 후 확정)."""
    df_clean = df.drop_duplicates().copy()
    removed = len(df) - len(df_clean)
    print(f"[중복 제거] {removed}행 제거")

    before_dropna = len(df_clean)
    df_clean = df_clean.dropna()
    removed_na = before_dropna - len(df_clean)
    print(f"[결측치 제거] {removed_na}행 제거")

    return df_clean.reset_index(drop=True)
