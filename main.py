# ---------------------------------------------------------------------------
# 작성자 : 문승인, 서수영, 이명로, 최승민, 정인제, 김서현
# 작성일 : 2026-07-21
# 작성목적 : UCI Adult 데이터셋 기반 데이터 분석 파이프라인
#            - 데이터 로딩(Pandas/Polars 비교) · EDA · 시각화 · 통계 분석 · ML Pipeline
#
# 본파일은 Skala 교육을 위한 Sample 코드이므로 작성자에게 모든 저작권이 있습니다.
#
# 변경사항 내역(날짜, 변경목적, 변경 내용 순으로 기입)
#   - 2026-07-21: 데이터 로딩 최적화 - Pandas 대비 Polars 적용을 통한 메모리 효율성 극대화 (약 80% 절감) |최승민|
#   - 2026-07-21: 데이터 전처리 - 중복 데이터(24행) 및 결측치(2,398행) 정제 프로세스 적용 |문승은|
#   - 2026-07-21: 피처 선택 - 상관계수(|r| >= 0.1) 및 카이제곱 검정(p < 0.05) 기반 유의미 변수 선별 |문승은|
#   - 2026-07-21: EDA 시각화 - Seaborn 히트맵 및 Plotly 그룹별 비교 박스플롯 자동 생성 및 저장 |김서현|
#   - 2026-07-21: 통계 분석 - 소득 그룹별 교육 기간 차이에 대한 독립표본 t-test 수행 및 유의성 검증 |정인제|
#   - 2026-07-21: ML 파이프라인 - RandomForest 분류 모델 구축 (Acc: 85.5%), .joblib 모델 저장 및 재로드 무결성 검증 |이명로|
#   - 2026-07-21: 리포트 자동화 - 분석 결과 종합 markdown(report.md) 자동 생성 모듈 연동 |서수영|
#
# ----------------------------------------------------------------------------

from src import (
    config,
    data_loader,
    eda,
    ml_pipeline,
    preprocessing,
    report,
    statistics_analysis,
    visualization,
)


def main() -> None:
    raw_path = data_loader.download_dataset()

    print("\n############# 데이터 로딩 (Pandas vs Polars) #############")
    df_pandas = data_loader.load_with_pandas(raw_path)
    df_polars = data_loader.load_with_polars(raw_path)
    data_loader.compare_pandas_polars(df_pandas, df_polars)
    raw_shape = df_pandas.shape

    print("\n############# 기본 EDA #############")
    eda.run_eda(df_pandas)

    print("\n############# 결측치 · 중복 처리 #############")
    preprocessing.check_missing(df_pandas)
    preprocessing.check_duplicates(df_pandas)
    df_clean = preprocessing.clean_data(df_pandas)

    print("\n############# 연관성 기반 피처 선택 #############")
    selected_numeric, selected_categorical, numeric_corr = (
        statistics_analysis.select_relevant_features(
            df_clean, config.NUMERIC_COLS, config.CATEGORICAL_COLS, config.TARGET_COL
        )
    )
    top_numeric_col = numeric_corr.abs().idxmax()

    print("\n############# 시각화 (Seaborn / Plotly) #############")
    heatmap_path = config.CHART_DIR / "correlation_heatmap.png"
    boxplot_path = config.CHART_DIR / "income_group_comparison.html"
    visualization.plot_correlation_heatmap(
        df_clean, selected_numeric, config.TARGET_COL, heatmap_path
    )
    visualization.plot_income_group_comparison(
        df_clean, config.TARGET_COL, top_numeric_col, boxplot_path
    )

    print("\n############# 통계 분석 #############")
    desc_stats = statistics_analysis.descriptive_statistics(df_clean, selected_numeric)
    corr_matrix = statistics_analysis.correlation_matrix(df_clean, selected_numeric)
    ttest_result = statistics_analysis.run_ttest(
        df_clean, config.TARGET_COL, top_numeric_col, "<=50K", ">50K"
    )
    ttest_info = (
        (top_numeric_col, "<=50K", ">50K", *ttest_result) if ttest_result else None
    )

    print("\n############# ML Pipeline (훈련 · 평가 · 저장 · 재로딩) #############")
    model_path = config.MODEL_DIR / "income_classifier.joblib"
    _, metrics = ml_pipeline.train_evaluate_save(
        df_clean, selected_numeric, selected_categorical, config.TARGET_COL, model_path
    )

    print("\n############# 리포트 자동 생성 #############")
    report.generate_report(
        raw_shape=raw_shape,
        clean_shape=df_clean.shape,
        selected_numeric=selected_numeric,
        selected_categorical=selected_categorical,
        desc_stats=desc_stats,
        corr_matrix=corr_matrix,
        ttest_info=ttest_info,
        metrics=metrics,
        chart_paths={"상관관계 히트맵": heatmap_path, "소득 그룹 비교": boxplot_path},
        model_path=model_path,
        report_path=config.REPORT_PATH,
    )


if __name__ == "__main__":
    main()
