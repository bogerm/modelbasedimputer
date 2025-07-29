import pandas as pd
import numpy as np
from modelbasedimputer import ModelBasedImputer

def test_basic_imputation():
    df = pd.DataFrame({
        "col1": ["A", "B", np.nan, "B", "A"],
        "col2": ["X", "Y", "X", "Y", np.nan],
        "num1": [1, 2, 3, np.nan, 5]
    })
    imputer = ModelBasedImputer(
        categorical_features=["col1", "col2"],
        numerical_features=["num1"]
    )
    imputer.fit(df)
    transformed = imputer.transform(df)
    assert not pd.isna(transformed).any().any()