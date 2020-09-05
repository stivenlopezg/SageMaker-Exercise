import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from config.config import GENDER_CAT, GEOGRAPHY_CAT


class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list):
        if not isinstance(columns, list):
            self.columns = [columns]
        else:
            self.columns = columns

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.loc[:, self.columns]
        return X


class ConvertDtypes(BaseEstimator, TransformerMixin):
    def __init__(self, numerical: list, categorical: list):
        if not isinstance(numerical, list):
            self.numerical = [numerical]
        else:
            self.numerical = numerical
        if not isinstance(categorical, list):
            self.categorical = [categorical]
        else:
            self.categorical = categorical

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        for numerical in self.numerical:
            X[numerical] = pd.to_numeric(X[numerical])
        for categorical in self.categorical:
            if categorical == 'Geography':
                categories = GEOGRAPHY_CAT
            else:
                categories = GENDER_CAT
            X[categorical] = pd.Categorical(X[categorical], categories=categories)
        return X


class GetDummies(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list):
        if not isinstance(columns, list):
            self.columns = [columns]
        else:
            self.columns = columns

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        for column in self.columns:
            if column == 'Gender':
                X = pd.get_dummies(X, columns=[column], drop_first=True)
            else:
                X = pd.get_dummies(X, columns=[column])
        return X
