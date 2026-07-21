import pandas as pd
from scipy import stats


def descriptive_statistics(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    desc = df[numeric_cols].agg(
        [
            "mean",
            "std",
            "median",
            lambda s: s.quantile(0.25),
            lambda s: s.quantile(0.75),
        ]
    )
    desc.index = ["mean", "std", "median", "q1(25%)", "q3(75%)"]

    print("=== 기술통계 (평균 · 표준편차 · 분위수) ===")
    print(desc)
    return desc


def correlation_matrix(df: pd.DataFrame, numeric_cols: list[str]) -> pd.DataFrame:
    corr = df[numeric_cols].corr()
    print("\n=== 수치형 변수 간 상관계수 ===")
    print(corr)
    return corr


def select_relevant_features(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    target_col: str,
    corr_threshold: float = 0.1,
    alpha: float = 0.05,
) -> tuple[list[str], list[str], pd.Series]:
    """income과 연관성이 있어 보이는 피처만 선정
    - 수치형: income과의 상관계수(|r| >= corr_threshold)
    - 범주형: 카이제곱 독립성 검정(p-value < alpha)
    """
    target_binary = pd.Series(pd.factorize(df[target_col])[0], index=df.index)

    numeric_corr = df[numeric_cols].corrwith(target_binary)
    selected_numeric = numeric_corr[numeric_corr.abs() >= corr_threshold].index.tolist()

    print(f"=== 수치형 변수 - income 상관계수 (|r| >= {corr_threshold} 선택) ===")
    print(numeric_corr.sort_values(key=abs, ascending=False))
    print(f"[선택된 수치형 변수] {selected_numeric}")

    selected_categorical = []
    print(f"\n=== 범주형 변수 - 카이제곱 검정 (p < {alpha} 선택) ===")
    for col in categorical_cols:
        contingency = pd.crosstab(df[col], df[target_col])
        _, p_value, _, _ = stats.chi2_contingency(contingency)
        mark = "선택" if p_value < alpha else "제외"
        print(f"[{col}] p-value={p_value:.6f} → {mark}")
        if p_value < alpha:
            selected_categorical.append(col)
    print(f"[선택된 범주형 변수] {selected_categorical}")

    return selected_numeric, selected_categorical, numeric_corr[selected_numeric]


def run_ttest(
    df: pd.DataFrame, group_col: str, value_col: str, group_a: str, group_b: str
) -> tuple[float, float] | None:
    """group_a/group_b 평균 차이를 Welch's t-test로 검정한다. report.md 자동 생성을 위해 (t_stat, p_value)를 반환한다."""
    a = df.loc[df[group_col] == group_a, value_col]
    b = df.loc[df[group_col] == group_b, value_col]

    if a.empty or b.empty:
        # 필터링 결과가 비어있으면 ttest_ind가 nan을 반환하므로 검정 자체를 건너뜀
        print(
            f"[t-test] '{group_a}' 또는 '{group_b}' 그룹 데이터가 없어 검정을 건너뜁니다."
        )
        return None

    t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)  # Welch's t-test

    print(f"\n=== t-test: {value_col} ({group_a} vs {group_b}) ===")
    print(f"[{group_a}] n={len(a)}, 평균={a.mean():.2f}")
    print(f"[{group_b}] n={len(b)}, 평균={b.mean():.2f}")
    print(f"t-statistic={t_stat:.4f}, p-value={p_value:.6f}")

    if p_value < 0.05:
        print("p < 0.05 → 두 그룹의 평균 차이는 통계적으로 유의미함")
    else:
        print("p >= 0.05 → 두 그룹의 평균 차이는 통계적으로 유의미하지 않음")

    return t_stat, p_value
