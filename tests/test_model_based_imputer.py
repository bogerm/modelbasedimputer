import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from modelbasedimputer import ModelBasedImputer  # Update this import path

@pytest.fixture
def sample_data():
    data = pd.DataFrame({
        'Stage_fear': ['Yes', 'No', np.nan, 'No', 'Yes'],
        'Drained_after_socializing': ['No', np.nan, 'Yes', 'No', np.nan],
        'Time_spent_Alone': [1.0, 2.5, 3.0, np.nan, 4.5],
        'Social_event_attendance': [np.nan, 7.0, 6.0, 5.0, 6.5],
    })
    return data

def test_fit_transform(sample_data):
    imputer = ModelBasedImputer(
        categorical_features=['Stage_fear', 'Drained_after_socializing'],
        numerical_features=['Time_spent_Alone', 'Social_event_attendance'],
        model_params={'max_iter': 200},
        verbose=True
    )

    imputer.set_output(transform="pandas")
    imputer.fit(sample_data)
    transformed = imputer.transform(sample_data)

    # Check that all missing values were imputed
    assert not transformed.isna().any().any(), "There are still missing values after transform"

    # Check column names preserved
    assert list(transformed.columns) == list(sample_data.columns), "Column names do not match original"

    # Check shape is preserved
    assert transformed.shape == sample_data.shape, "Output shape doesn't match input"

def test_supports_numerical_and_categorical(sample_data):
    imputer = ModelBasedImputer(
        categorical_features=['Stage_fear'],
        numerical_features=['Time_spent_Alone'],
        verbose=True
    )

    imputer.set_output(transform="pandas")
    imputer.fit(sample_data)
    transformed = imputer.transform(sample_data)

    assert 'Stage_fear' in transformed.columns
    assert 'Time_spent_Alone' in transformed.columns
    assert transformed.isna().sum().sum() == 0, "Imputer failed to fill all NaNs"

def test_transform_works_after_fit(sample_data):
    imputer = ModelBasedImputer(
        categorical_features=['Stage_fear'],
        model_class=LogisticRegression
    )

    imputer.set_output(transform="pandas")
    imputer.fit(sample_data)
    transformed = imputer.transform(sample_data.copy())
    assert isinstance(transformed, pd.DataFrame)

def test_handles_no_missing_values():
    df = pd.DataFrame({
        'Stage_fear': ['Yes', 'No', 'Yes'],
        'Time_spent_Alone': [1.0, 2.0, 3.0]
    })

    imputer = ModelBasedImputer(
        categorical_features=['Stage_fear'],
        numerical_features=['Time_spent_Alone'],
        #model_class=LogisticRegression
    )

    imputer.set_output(transform="pandas")
    imputer.fit(df)
    transformed = imputer.transform(df)

    assert transformed.equals(df), "Imputer altered data with no missing values"

