import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.utils.validation import check_is_fitted
from sklearn.ensemble import HistGradientBoostingClassifier

from modelbasedimputer import ModelBasedImputer  # adjust the import path

@pytest.fixture
def sample_data():
    np.random.seed(42)
    df = pd.DataFrame({
        'Stage_fear': np.random.choice(['Yes', 'No'], size=100),
        'Drained_after_socializing': np.random.choice(['Yes', 'No'], size=100),
        'Time_spent_Alone': np.random.normal(5, 2, size=100),
        'Going_outside': np.random.normal(3, 1, size=100),
    })

    # Inject missing values
    df.loc[df.sample(frac=0.1).index, 'Stage_fear'] = np.nan
    df.loc[df.sample(frac=0.1).index, 'Time_spent_Alone'] = np.nan

    return df

def test_fit_and_transform_onehot(sample_data):
    imputer = ModelBasedImputer(
        categorical_features=['Stage_fear', 'Drained_after_socializing'],
        numerical_features=['Time_spent_Alone', 'Going_outside'],
        model_class=HistGradientBoostingClassifier,
        num_model_class=Ridge,
        encoder_type='onehot',
        verbose=True
    ).set_output(transform="pandas")

    imputer.fit(sample_data)
    transformed = imputer.transform(sample_data)

    assert isinstance(transformed, (pd.DataFrame, np.ndarray))
    if isinstance(transformed, pd.DataFrame):
        assert not transformed.isna().any().any(), "There are still missing values after transform"
    else:
        assert not np.isnan(transformed).any(), "There are still missing values in NumPy output"

def test_fit_and_transform_target(sample_data):
    imputer = ModelBasedImputer(
        categorical_features=['Stage_fear'],
        numerical_features=[],
        model_class=HistGradientBoostingClassifier,
        encoder_type='target',
    ).set_output(transform="pandas")

    imputer.fit(sample_data)
    transformed = imputer.transform(sample_data)

    assert transformed.shape[0] == sample_data.shape[0]
    assert not transformed['Stage_fear'].isna().any()

def test_set_output_pandas(sample_data):
    imputer = ModelBasedImputer(
        categorical_features=['Stage_fear'],
        numerical_features=['Time_spent_Alone'],
        encoder_type='onehot'
    ).set_output(transform='pandas')

    imputer.fit(sample_data)
    transformed = imputer.transform(sample_data)
    assert isinstance(transformed, pd.DataFrame)
    assert set(transformed.columns) == set(sample_data.columns)

def test_check_is_fitted(sample_data):
    imputer = ModelBasedImputer(
        categorical_features=['Stage_fear'],
        numerical_features=['Time_spent_Alone']
    ).set_output(transform="pandas")

    with pytest.raises(Exception):
        imputer.transform(sample_data)

    imputer.fit(sample_data)
    imputer.transform(sample_data)  # Should not raise

def test_transform_without_missing(sample_data):
    imputer = ModelBasedImputer(
        categorical_features=['Drained_after_socializing'],
        numerical_features=['Going_outside']
    ).set_output(transform="pandas")
    df_no_missing = sample_data.dropna().copy()
    imputer.fit(df_no_missing)
    transformed = imputer.transform(df_no_missing)

    assert transformed.shape == df_no_missing.shape

def test_transform_with_missing_columns():
    df = pd.DataFrame({
        "A": ["yes", "no", "yes", np.nan],
        "B": [1.0, 2.0, 3.0, 4.0],
    })

    imputer = ModelBasedImputer(
        categorical_features=["A"],
        numerical_features=["B"]
    ).fit(df)

    # Drop column B
    df_missing = df.drop(columns=["B"])

    with pytest.raises(ValueError, match="Input is missing expected columns"):
        imputer.transform(df_missing)

def test_encoder_types(monkeypatch):
    df = pd.DataFrame({
        "cat1": ["a", "b", np.nan, "a", "b", "b", "a", "a"],
        "cat2": ["x", "y", "x", "y", "x", "x", "y", "y"]
    })

    # Force a NaN to be imputed
    df.loc[2, "cat1"] = np.nan

    # OneHotEncoder test
    imputer_onehot = ModelBasedImputer(
        categorical_features=["cat1"],
        encoder_type="onehot"
    ).fit(df)
    result_onehot = imputer_onehot.transform(df)
    assert not pd.DataFrame(result_onehot).isna().any().any()

    # TargetEncoder test
    imputer_target = ModelBasedImputer(
        categorical_features=["cat1"],
        encoder_type="target"
    ).fit(df)
    result_target = imputer_target.transform(df)
    assert not pd.DataFrame(result_target).isna().any().any()

def test_inverse_transform_with_label_encoder():
    df = pd.DataFrame({
        "animal": ["cat", "dog", "cat", np.nan, "dog", np.nan],
        "age": [1, 2, 3, 4, 5, 6]
    })

    imputer = ModelBasedImputer(
        categorical_features=["animal"],
        numerical_features=[],
        verbose=True
    ).fit(df)

    result = pd.DataFrame(imputer.transform(df), columns=imputer.all_features_)
    assert not result["animal"].isna().any()
    assert set(result["animal"]).issubset({"cat", "dog"})

def test_pipeline_preserves_shape_and_type():
    df = pd.DataFrame({
        "city": ["NY", "LA", "SF", np.nan],
        "salary": [100_000, 120_000, 110_000, np.nan]
    })

    imputer = ModelBasedImputer(
        categorical_features=["city"],
        numerical_features=["salary"],
        verbose=False
    ).fit(df)

    result = imputer.transform(df)
    assert result.shape == df.shape
    assert isinstance(result, (pd.DataFrame, np.ndarray))