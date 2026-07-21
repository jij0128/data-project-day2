# ---------------------------------------------------------------------------
# 작성자 : 문승인, 서수영, 이명로, 최승민, 정인제, 김서현
# 작성일 : 2026-07-21
# 작성목적 : ColumnTransformer + RandomForestClassifier Pipeline 구성, 학습·평가·저장·재로딩
#
# 본파일은 Skala 교육을 위한 Sample 코드이므로 작성자에게 모든 저작권이 있습니다.
#
# 변경사항 내역(날짜, 변경목적, 변경 내용 순으로 기입)
#   - 2026-07-21, report.md 자동 생성 지원, train_evaluate_save가 평가지표 dict를 반환하도록 수정
#
# ----------------------------------------------------------------------------

from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_pipeline(numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    """수치형 표준화 + 범주형 원-핫 인코딩(ColumnTransformer)과 RandomForestClassifier를 하나의 Pipeline으로 구성한다."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200, max_depth=12, random_state=42, n_jobs=-1
                ),
            ),
        ]
    )


def train_evaluate_save(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    target_col: str,
    model_path: Path,
) -> tuple[Pipeline, dict[str, float]]:
    """Pipeline을 학습·평가하고 accuracy/precision/recall/F1을 계산한 뒤 joblib으로 저장, 재로딩까지 검증한다."""
    X = df[numeric_cols + categorical_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline(numeric_cols, categorical_cols)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    pos_label = ">50K"
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, pos_label=pos_label),
        "recall": recall_score(y_test, y_pred, pos_label=pos_label),
        "f1": f1_score(y_test, y_pred, pos_label=pos_label),
    }
    print("=== 평가 지표 ===")
    for name, value in metrics.items():
        print(f"{name:<9}={value:.4f}")
    print("\n=== classification report ===")
    print(classification_report(y_test, y_pred))

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"[저장] 모델을 저장했습니다: {model_path}")

    # 재로딩 후 예측이 동일한지 확인
    reloaded_pipeline = joblib.load(model_path)
    y_pred_reloaded = reloaded_pipeline.predict(X_test)
    print(
        f"[재로딩] 재로딩한 모델의 예측이 기존과 동일함: {(y_pred == y_pred_reloaded).all()}"
    )

    return pipeline, metrics
