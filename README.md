# ModelBasedImputer

**ModelBasedImputer** is a scikit-learn-compatible transformer that imputes missing values in both categorical and numerical features using predictive models. It trains a separate model for each column with missing data, leveraging the remaining available data as features.

---

## Features

✅ Impute missing categorical and numerical values using supervised learning  
✅ Supports one-hot and target encoding  
✅ Customizable models per column (e.g., `HistGradientBoostingClassifier`, `Ridge`, etc.)  
✅ Optional fallback to simple statistics (mean/mode)  
✅ scikit-learn compatible with `Pipeline` and `ColumnTransformer`  
✅ Verbose logging for easy debugging  
✅ Built-in `set_output(transform="pandas")` support

---

## Installation

```bash
git clone https://github.com/your-username/ModelBasedImputer.git
cd ModelBasedImputer
pip install -e .
```
#### Make sure the following dependencies are installed:
```bash
pip install scikit-learn category_encoders pandas numpy
```

## Usage Example

```python
from model_based_imputer import ModelBasedImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import Ridge
import pandas as pd

# Sample data
df = pd.read_csv("your_dataset.csv")

imputer = ModelBasedImputer(
    categorical_features=["education", "workclass"],
    numerical_features=["age", "hours-per-week"],
    model_class=HistGradientBoostingClassifier,
    num_model_class=Ridge,
    encoder_type="target",
    verbose=True,
    fallback=True
)

imputer.set_output(transform="pandas")
df_imputed = imputer.fit_transform(df)
```

## Parameters
| Parameter              | Type            | Description                             |
| ---------------------- | --------------- | --------------------------------------- |
| `categorical_features` | `List[str]`     | List of categorical columns to impute   |
| `numerical_features`   | `List[str]`     | List of numerical columns to impute     |
| `model_class`          | `BaseEstimator` | Model for categorical features          |
| `num_model_class`      | `BaseEstimator` | Model for numerical features            |
| `model_params`         | `dict`          | Parameters for `model_class`            |
| `num_model_params`     | `dict`          | Parameters for `num_model_class`        |
| `encoder_type`         | `str`           | `"onehot"` or `"target"`                |
| `verbose`              | `bool`          | Print fit/transform logs                |
| `fallback`             | `bool`          | Use mode/mean if model can't be trained |

## Tests
Run the unit and functional tests using:
```bash
pytest tests/
```

## Limitations
- For very sparse datasets, model training may be skipped (with fallback enabled).
- target encoding may introduce leakage if not used carefully (e.g., during cross-validation).

## License
MIT License. See LICENSE file for details.

## Author
Developed by Oleg Buklanov – feel free to contribute or open issues!