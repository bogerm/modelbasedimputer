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
    assert not pd.DataFrame(transformed).isna().any().any()

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
        check_is_fitted(imputer)

    imputer.fit(sample_data)
    check_is_fitted(imputer)  # Should not raise

def test_transform_without_missing(sample_data):
    imputer = ModelBasedImputer(
        categorical_features=['Drained_after_socializing'],
        numerical_features=['Going_outside']
    ).set_output(transform="pandas")
    df_no_missing = sample_data.dropna().copy()
    imputer.fit(df_no_missing)
    transformed = imputer.transform(df_no_missing)

    assert transformed.shape == df_no_missing.shape
