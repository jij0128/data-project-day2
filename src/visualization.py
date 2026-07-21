from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns

# 그래프에서 한글이 깨지지 않도록 폰트 설정 (macOS 기준)
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False


def plot_correlation_heatmap(df: pd.DataFrame, numeric_cols: list[str], target_col: str, output_path: Path) -> None:
    """Seaborn 정적 차트 - income과 유의미하게 연관된 수치형 변수만 골라 상관관계 히트맵(income 포함)"""
    target_binary = pd.Series(pd.factorize(df[target_col])[0], index=df.index, name=target_col)
    corr = pd.concat([df[numeric_cols], target_binary], axis=1).corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("income과 유의미하게 연관된 변수 상관관계 히트맵")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    print(f"[저장] Seaborn 상관관계 히트맵: {output_path}")
    plt.show()


def plot_income_group_comparison(df: pd.DataFrame, target_col: str, numeric_col: str, output_path: Path) -> None:
    """Plotly 인터랙티브 차트 - income과 가장 강하게 연관된 수치형 변수를 소득 그룹별로 비교(박스플롯)"""
    fig = px.box(
        df,
        x=target_col,
        y=numeric_col,
        color=target_col,
        title=f"소득 그룹별 {numeric_col} 비교 (income과 상관관계가 가장 높은 변수)",
        labels={target_col: "소득 구간"},
    )
    fig.update_layout(legend_title_text="소득 구간")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output_path)
    print(f"[저장] Plotly 그룹 비교 차트: {output_path}")
    fig.show()
