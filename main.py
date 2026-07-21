# ---------------------------------------------------------------------------
# 작성자 : 서수영
# 작성일 : 2026-07-21
# 작성목적 : UCI Adult 데이터셋 기반 데이터 분석 파이프라인
#            - 데이터 로딩(Pandas/Polars 비교) · EDA · 시각화 · 통계 분석 · ML Pipeline
#
# 본파일은 Skala 교육을 위한 Sample 코드이므로 작성자에게 모든 저작권이 있습니다.
#
# 변경사항 내역(날짜, 변경목적, 변경 내용 순으로 기입)
#
# ----------------------------------------------------------------------------

from src import config, data_loader, eda, ml_pipeline, preprocessing, statistics_analysis, visualization


def main() -> None:
    raw_path = data_loader.download_dataset()

    print("\n############# 데이터 로딩 (Pandas vs Polars) #############")
    df_pandas = data_loader.load_with_pandas(raw_path)
    df_polars = data_loader.load_with_polars(raw_path)
    data_loader.compare_pandas_polars(df_pandas, df_polars)

    print("\n############# 기본 EDA #############")
    eda.run_eda(df_pandas)

    print("\n############# 결측치 · 중복 처리 #############")
    preprocessing.check_missing(df_pandas)
    preprocessing.check_duplicates(df_pandas)
    df_clean = preprocessing.clean_data(df_pandas)

    print("\n############# 연관성 기반 피처 선택 #############")
    selected_numeric, selected_categorical, numeric_corr = statistics_analysis.select_relevant_features(
        df_clean, config.NUMERIC_COLS, config.CATEGORICAL_COLS, config.TARGET_COL
    )
    top_numeric_col = numeric_corr.abs().idxmax()

    print("\n############# 시각화 (Seaborn / Plotly) #############")
    visualization.plot_correlation_heatmap(
        df_clean, selected_numeric, config.TARGET_COL, config.CHART_DIR / "correlation_heatmap.png"
    )
    visualization.plot_income_group_comparison(
        df_clean, config.TARGET_COL, top_numeric_col, config.CHART_DIR / "income_group_comparison.html"
    )

    print("\n############# 통계 분석 #############")
    statistics_analysis.descriptive_statistics(df_clean, selected_numeric)
    statistics_analysis.correlation_matrix(df_clean, selected_numeric)
    statistics_analysis.run_ttest(df_clean, config.TARGET_COL, top_numeric_col, "<=50K", ">50K")

    print("\n############# ML Pipeline (훈련 · 평가 · 저장 · 재로딩) #############")
    ml_pipeline.train_evaluate_save(
        df_clean,
        selected_numeric,
        selected_categorical,
        config.TARGET_COL,
        config.MODEL_DIR / "income_classifier.joblib",
    )


if __name__ == "__main__":
    main()
