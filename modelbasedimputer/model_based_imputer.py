from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils.validation import check_is_fitted
from category_encoders import TargetEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
import pandas as pd
import numpy as np


class ModelBasedImputer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        categorical_features=None,
        numerical_features=None,
        model_class=HistGradientBoostingClassifier,
        num_model_class=Ridge,
        model_params=None,
        num_model_params=None,
        encoder_type='onehot',  # 'onehot' or 'target'
        verbose=False
    ):
        self.categorical_features = categorical_features or []
        self.numerical_features = numerical_features or []
        self.model_class = model_class
        self.num_model_class = num_model_class
        self.model_params = model_params or {}
        self.num_model_params = num_model_params or {}
        self.encoder_type = encoder_type
        self.verbose = verbose

        self.models_ = {}
        self.label_encoders_ = {}
        self.encoders_ = {}
        self.all_features_ = None
        self._output_config = None

        if self.verbose and hasattr(self.model_class, 'predict_proba') is False:
            print("Warning: selected model may not support missing values.")

    def set_output(self, *, transform=None):
        self._output_config = transform
        return self

    def fit(self, X, y=None):
        X = pd.DataFrame(X).copy()
        self.all_features_ = X.columns.tolist()

        if self.verbose:
            print("Starting fit...")

        # Fit models for categorical columns
        for col in self.categorical_features:
            X_train = X.dropna(subset=[col])
            X_train = X_train[X_train[self.categorical_features + self.numerical_features].isna().sum(axis=1) == 0]
            if X_train.empty:
                continue

            le = LabelEncoder()
            y_train = le.fit_transform(X_train[col])
            self.label_encoders_[col] = le

            feature_cols = [c for c in X.columns if c != col]
            X_train_features = X_train[feature_cols]

            # Encoder selection
            if self.encoder_type == 'onehot':
                encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
                X_encoded = encoder.fit_transform(X_train_features)
            elif self.encoder_type == 'target':
                encoder = TargetEncoder()
                X_encoded = encoder.fit_transform(X_train_features, y_train)
            else:
                raise ValueError("encoder_type must be 'onehot' or 'target'")

            self.encoders_[col] = encoder

            model = self.model_class(**self.model_params)
            model.fit(X_encoded, y_train)
            self.models_[col] = model

            if self.verbose:
                print(f"Fitted model for categorical feature: {col}")

        # Fit models for numerical columns
        for col in self.numerical_features:
            X_train = X.dropna(subset=[col])
            X_train = X_train[X_train[self.categorical_features + self.numerical_features].isna().sum(axis=1) == 0]
            if X_train.empty:
                continue

            y_train = X_train[col]
            feature_cols = [c for c in X.columns if c != col]
            X_train_features = X_train[feature_cols]

            # Always use OneHotEncoder for numerical
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            X_encoded = encoder.fit_transform(X_train_features)

            self.encoders_[col] = encoder

            model = self.num_model_class(**self.num_model_params)
            model.fit(X_encoded, y_train)
            self.models_[col] = model

            if self.verbose:
                print(f"Fitted model for numerical feature: {col}")

        return self

    def transform(self, X):
        check_is_fitted(self, ["models_", "all_features_"])
        X = pd.DataFrame(X).copy()

        if self.verbose:
            print("Starting transform...")

        for col in self.models_:
            model = self.models_[col]
            encoder = self.encoders_[col]
            is_cat = col in self.categorical_features
            le = self.label_encoders_.get(col, None)

            missing_mask = X[col].isna()
            valid_mask = missing_mask

            if valid_mask.sum() == 0:
                continue

            feature_cols = [c for c in X.columns if c != col]
            X_pred = X.loc[valid_mask, feature_cols]

            try:
                X_encoded = encoder.transform(X_pred)
                y_pred = model.predict(X_encoded)
                if is_cat:
                    y_pred = le.inverse_transform(np.round(y_pred).astype(int))
                X.loc[valid_mask, col] = y_pred
                if self.verbose:
                    print(f"Imputed {valid_mask.sum()} missing values in '{col}'")
            except Exception as e:
                if self.verbose:
                    print(f"Skipping column '{col}' due to error: {e}")

        result = X[self.all_features_]
        if self._output_config == "pandas":
            return pd.DataFrame(result, columns=self.all_features_, index=X.index)
        return result.values