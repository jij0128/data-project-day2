# ---------------------------------------------------------------------------
# 작성자 : 문승인, 서수영, 이명로, 최승민, 정인제, 김서현
# 작성일 : 2026-07-21
# 작성목적 : 데이터 준비~ML Pipeline 결과를 report.md로 자동 생성
#
# 본파일은 Skala 교육을 위한 Sample 코드이므로 작성자에게 모든 저작권이 있습니다.
#
# 변경사항 내역(날짜, 변경목적, 변경 내용 순으로 기입)
#   - 2026-07-21, 자동화 요구사항 충족, 신규 작성
#
# ----------------------------------------------------------------------------

from pathlib import Path

import pandas as pd


def _to_markdown_table(df: pd.DataFrame) -> str:
    """tabulate 의존성 없이 DataFrame을 마크다운 표 문자열로 변환한다."""
    df = df.reset_index()
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    separator = "|" + "|".join([" --- "] * len(df.columns)) + "|"
    rows = [
        "| "
        + " | ".join(f"{v:.2f}" if isinstance(v, float) else str(v) for v in row)
        + " |"
        for row in df.itertuples(index=False)
    ]
    return "\n".join([header, separator, *rows])


def generate_report(
    raw_shape: tuple[int, int],
    clean_shape: tuple[int, int],
    selected_numeric: list[str],
    selected_categorical: list[str],
    desc_stats: pd.DataFrame,
    corr_matrix: pd.DataFrame,
    ttest_info: tuple[str, str, str, float, float] | None,
    metrics: dict[str, float],
    chart_paths: dict[str, Path],
    model_path: Path,
    report_path: Path,
) -> None:
    """데이터 준비부터 ML Pipeline까지의 분석 결과를 report.md로 자동 생성한다."""
    desc_md = _to_markdown_table(desc_stats)
    corr_md = _to_markdown_table(corr_matrix)

    if ttest_info is not None:
        value_col, group_a, group_b, t_stat, p_value = ttest_info
        verdict = "유의미한 차이 있음" if p_value < 0.05 else "유의미한 차이 없음"
        ttest_section = (
            f"- 대상: `{value_col}` ({group_a} vs {group_b})\n"
            f"- t-statistic={t_stat:.4f}, p-value={p_value:.6f}\n"
            f"- 해석: {verdict} (p<0.05 기준)"
        )
    else:
        ttest_section = "- 그룹 데이터 부족으로 t-test를 수행하지 못했습니다."

    chart_lines = "\n".join(
        f"- [{name}]({path.name})" for name, path in chart_paths.items()
    )
    metric_lines = "\n".join(
        f"- {name}: {value:.4f}" for name, value in metrics.items()
    )

    content = f"""# End2End 데이터 분석 리포트 - Adult Census Income

## 1. 데이터 준비
- 원본 shape: {raw_shape}
- 정제 후 shape: {clean_shape} (결측치·중복 제거)
- 선택된 수치형 변수: {selected_numeric}
- 선택된 범주형 변수: {selected_categorical}

## 2. 시각화
{chart_lines}

## 3. 통계 분석
### 기술통계 (평균·표준편차·분위수)
{desc_md}

### 상관계수
{corr_md}

### t-test
{ttest_section}

## 4. ML Pipeline
{metric_lines}
- 저장된 모델 파일: {model_path.name}
"""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(content, encoding="utf-8")
    print(f"[저장] report.md 자동 생성 완료: {report_path}")
