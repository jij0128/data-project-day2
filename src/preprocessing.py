# ---------------------------------------------------------------------------
# 작성자 : 문승인, 서수영, 이명로, 최승민, 정인제, 김서현
# 작성일 : 2026-07-21
# 작성목적 : 결측치·중복 행 현황 확인 및 제거
#
# 본파일은 Skala 교육을 위한 Sample 코드이므로 작성자에게 모든 저작권이 있습니다.
#
# 변경사항 내역(날짜, 변경목적, 변경 내용 순으로 기입)
#   - 2026-07-21, 토의를 통한 결측치 처리 방식 수정, 최빈값 대체 대신 dropna로 제거하도록 변경
#   - 2026-07-21, 완성도(주석) 보완, 각 함수에 docstring 추가
#
# ----------------------------------------------------------------------------

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
