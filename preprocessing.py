"""Shared preprocessing pipeline (fit on train only, reused at prediction time)."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from generate_dataset import FEATURE_COLUMNS


def build_preprocessor(
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> ColumnTransformer:
    """Build a ColumnTransformer that is safe to persist with joblib.

    Current synthetic table is fully numeric. Categorical branches are included
    so extra lifestyle/wearable categories can be added later without rewriting
    the training script.
    """
    numeric_features = numeric_features or list(FEATURE_COLUMNS)
    categorical_features = categorical_features or []

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    transformers = [("numeric", numeric_pipeline, numeric_features)]

    if categorical_features:
        from sklearn.preprocessing import OneHotEncoder

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_features))

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_model_pipeline(estimator, preprocessor: ColumnTransformer | None = None) -> Pipeline:
    """Wrap preprocessor + classifier so train and serve use the same steps."""
    return Pipeline(
        steps=[
            ("preprocess", preprocessor or build_preprocessor()),
            ("model", estimator),
        ]
    )
