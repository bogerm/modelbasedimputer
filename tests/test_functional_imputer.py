import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, r2_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import Ridge
from your_module.model_based_imputer import ModelBasedImputer  # adjust as needed


def test_model_based_imputer_functional(seed=42):
    np.random.seed(seed)

    # 1. Create synthetic data
    n_samples = 500
    data = pd.DataFrame({
        "cat1": np.random.choice(["A", "B", "C"], size=n_samples),
        "cat2": np.random.choice(["X", "Y"], size=n_samples),
        "num1": np.random.normal(10, 2, size=n_samples),
        "num2": np.random.uniform(0, 1, size=n_samples)
    })

    # Save original full dataset for later comparison
    original_data = data.copy(deep=True)

    # 2. Randomly mask 25% of values in selected columns
    mask_fraction = 0.25
    mask = pd.DataFrame(False, index=data.index, columns=data.columns)

    for col in data.columns:
        mask_indices = data.sample(frac=mask_fraction).index
        data.loc[mask_indices, col] = np.nan
        mask.loc[mask_indices, col] = True

    # 3. Impute using ModelBasedImputer
    imputer = ModelBasedImputer(
        categorical_features=["cat1", "cat2"],
        numerical_features=["num1", "num2"],
        model_class=HistGradientBoostingClassifier,
        num_model_class=Ridge,
        encoder_type="onehot",
        verbose=False
    )
    imputer.fit(data)
    imputed_data = imputer.transform(data)

    # 4. Evaluate accuracy and R² score
    cat_acc = {}
    num_r2 = {}

    for col in data.columns:
        true_values = original_data.loc[mask[col], col]
        pred_values = imputed_data[mask[col]][col]

        if col.startswith("cat"):
            acc = accuracy_score(true_values, pred_values)
            cat_acc[col] = acc
        else:
            r2 = r2_score(true_values, pred_values)
            num_r2[col] = r2

    # 5. Print results
    print("Categorical Accuracy:")
    for col, acc in cat_acc.items():
        print(f"  {col}: {acc:.4f}")

    print("\nNumerical R² Score:")
    for col, r2 in num_r2.items():
        print(f"  {col}: {r2:.4f}")

    # Optional asserts
    assert all(acc > 0.7 for acc in cat_acc.values()), "Low categorical accuracy"
    assert all(r2 > 0.7 for r2 in num_r2.values()), "Low numerical R²"

