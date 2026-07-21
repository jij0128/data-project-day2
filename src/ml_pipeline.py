from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_pipeline(numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)),
        ]
    )


def train_evaluate_save(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
    target_col: str,
    model_path: Path,
) -> Pipeline:
    X = df[numeric_cols + categorical_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline(numeric_cols, categorical_cols)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    pos_label = ">50K"
    print("=== 평가 지표 ===")
    print(f"accuracy ={accuracy_score(y_test, y_pred):.4f}")
    print(f"precision={precision_score(y_test, y_pred, pos_label=pos_label):.4f}")
    print(f"recall   ={recall_score(y_test, y_pred, pos_label=pos_label):.4f}")
    print(f"f1       ={f1_score(y_test, y_pred, pos_label=pos_label):.4f}")
    print("\n=== classification report ===")
    print(classification_report(y_test, y_pred))

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"[저장] 모델을 저장했습니다: {model_path}")

    # 재로딩 후 예측이 동일한지 확인
    reloaded_pipeline = joblib.load(model_path)
    y_pred_reloaded = reloaded_pipeline.predict(X_test)
    print(f"[재로딩] 재로딩한 모델의 예측이 기존과 동일함: {(y_pred == y_pred_reloaded).all()}")

    return pipeline
