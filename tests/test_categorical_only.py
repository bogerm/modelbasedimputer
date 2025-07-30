import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import LabelEncoder

from modelbasedimputer import ModelBasedImputer


# @pytest.mark.functional
@pytest.mark.skip(reason="Temporarily disabled")
def test_model_based_imputer_categorical_only_simpleimputer_numerical():
    # 1. Load dataset
    df = fetch_openml("adult", version=2, as_frame=True)["data"]
    df = df[["education", "workclass", "occupation", "hours-per-week", "age", "capital-gain"]].copy()
    df = df.dropna()

    # 2. Encode categorical columns
    categorical_cols = ["education", "workclass", "occupation"]
    numerical_cols = ["hours-per-week", "age", "capital-gain"]
    df_encoded = df.copy()
    label_encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # 3. Introduce missing values (25%)
    df_missing = df_encoded.copy()
    mask = np.random.rand(*df_missing.shape) < 0.25
    df_missing_masked = df_missing.mask(mask)

    # 4. Backup original values for scoring
    ground_truth = df_encoded.copy()
    missing_mask = df_missing_masked.isna()

    # 5. Impute:
    #    - ModelBasedImputer for categorical
    #    - SimpleImputer for numerical
    cat_imputer = ModelBasedImputer(
        categorical_features=categorical_cols,
        numerical_features=[],  # disable numerical
        encoder_type="target",
        verbose=True,
    ).set_output(transform="pandas")
    df_cat_imputed = cat_imputer.fit_transform(df_missing_masked)

    # SimpleImputer for numerical
    # num_imputer = SimpleImputer(strategy="mean")
    num_imputer = KNNImputer(n_neighbors=5)

    df_num_imputed = pd.DataFrame(
        num_imputer.fit_transform(df_missing_masked[numerical_cols]),
        columns=numerical_cols,
        index=df_missing_masked.index,
    )

    # Combine both
    imputed_df = df_cat_imputed.copy()
    for col in numerical_cols:
        imputed_df[col] = df_num_imputed[col]

    # 6. Score accuracy
    cat_scores = {}
    for col in categorical_cols:
        missing = missing_mask[col]
        if missing.sum() > 0:
            score = accuracy_score(
                ground_truth.loc[missing, col],
                imputed_df.loc[missing, col],
            )
            cat_scores[col] = score

    num_scores = {}
    for col in numerical_cols:
        missing = missing_mask[col]
        if missing.sum() > 0:
            score = r2_score(
                ground_truth.loc[missing, col],
                imputed_df.loc[missing, col],
            )
            num_scores[col] = score

    # 7. Print summary
    print("\nCategorical Accuracy:")
    for col, score in cat_scores.items():
        print(f"{col}: {score:.3f}")
    print("\nNumerical R²:")
    for col, score in num_scores.items():
        print(f"{col}: {score:.3f}")

    # 8. Assertions
    assert all(score > 0.0 for score in cat_scores.values()), "Low categorical accuracy"
    # assert all(score > 0.0 for score in num_scores.values()), "Low numerical R²"
