import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from modelbasedimputer import ModelBasedImputer  # Replace with actual path

def test_model_based_imputer_on_realistic_data(seed=42):
    np.random.seed(seed)

    # 1. Load Adult dataset (mixed types)
    # adult = fetch_openml(name="adult", version=2, as_frame=True)
    # df = adult.frame
    personality = pd.read_csv('tests/personality_dataset.csv')
    df = personality.drop(columns=['Personality'])

    # 2. Keep relevant subset of features
    # features = [
    #     "education", "workclass", "occupation",
    #     "hours-per-week", "age", "capital-gain", "education-num", "marital-status"
    # ]
    features = df.columns.tolist()
    df = df[features].copy()
    df = df.dropna()  # Drop initial NaNs

    # 3. Randomly remove 25% of values in each column
    mask_fraction = 0.20
    missing_mask = pd.DataFrame(False, index=df.index, columns=df.columns)
    for col in df.columns:
        mask_idx = df.sample(frac=mask_fraction).index
        df.loc[mask_idx, col] = np.nan
        missing_mask.loc[mask_idx, col] = True

    original = personality[features].loc[df.index]  # Original data before masking

    # 4. Define types
    # categorical = ["education", "workclass", "occupation"]
    # numerical = ["hours-per-week", "age", "capital-gain"]
    numerical = df.select_dtypes(include=["number"]).columns.tolist()
    categorical = df.select_dtypes(include=["object"]).columns.tolist()

    # 5. Impute
    imputer = ModelBasedImputer(
        categorical_features=categorical,
        numerical_features=numerical,
        encoder_type="target",
        model_class=HistGradientBoostingRegressor,
        verbose=True
    ).set_output(transform="pandas")
    imputer.fit(df)
    imputed = imputer.transform(df)

    # 6. Score imputation quality
    cat_scores = {}
    num_scores = {}
    for col in df.columns:
        true = original.loc[missing_mask[col], col]
        pred = imputed.loc[missing_mask[col], col]

        if pred.isna().any():
            print(f"Warning: NaNs still present in predictions for '{col}'")
            continue  # or raise

        if col in categorical:
            score = accuracy_score(true, pred)
            cat_scores[col] = score
        else:
            score = r2_score(true, pred,)
            num_scores[col] = score

    print("\nCategorical Accuracy:")
    for col, score in cat_scores.items():
        print(f"{col}: {score:.3f}")

    print("\nNumerical R²:")
    for col, score in num_scores.items():
        print(f"{col}: {score:.3f}")

    # Assert acceptable performance
    assert all(score > 0.6 for score in cat_scores.values()), "Low categorical accuracy"
    assert all(score > 0.5 for score in num_scores.values()), "Low numerical R²"

