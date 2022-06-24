import os
import pickle

import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression


class Model:
    """API for training and saving the model."""

    def __init__(self, model_type: str = "xgb"):
        valid_model_types = ["xgb", "logistic"]
        assert model_type in valid_model_types, f"Model type must be one of {valid_model_types}"

        if model_type == "xgb":
            self.model = xgb.XGBClassifier()  # (n_estimators=100, max_depth=3, learning_rate=0.1)
        elif model_type == "logistic":
            self.model = LogisticRegression(max_iter=1000)

        self.model_type = model_type
        self.fitted = False
        self.feature_names = None
        self.target_names = None

    def set_feature_names(self, feature_names: list):
        """
        Set the feature names.
        :param feature_names: List of feature names.
        """
        self.feature_names = feature_names

    def set_target_names(self, target_names: dict):
        """
        Set the target names.
        :param target_names: Dict with the target names.
        """
        self.target_names = target_names

    def train(self, X: np.ndarray, y: np.ndarray):
        """Fit the model to the training data.
        :param X: Features.
        :param y: Labels.
        """
        self.model.fit(X, y)
        self.fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the labels for the given features.
        :param X: Features.
        :return: Predicted labels."""
        assert self.fitted, "Model must be fitted before predicting."
        predictions = self.model.predict(X)
        return predictions

    def save(self, path: str = "model_directory/model.pkl"):
        """Save the model object to the given path.
        :param path: Path to save the model to.
        """
        assert self.fitted, "Model must be fitted before saving."
        # make sure the directory exists
        directory_path = "/".join(path.split("/")[:-1])
        os.makedirs(directory_path, exist_ok=True)
        # save the object
        pickle.dump(self, open(path, "wb"))

    def load(self, path: str = "model_directory/model.pkl") -> "Model":
        """Load the Model object from the given path.
        :param path: Path to load the model from.
        :return: Model object."""
        return pickle.load(open(path, "rb"))

    def print_features_importance(self):
        """Get the feature importance of the model."""
        assert self.fitted, "Model must be fitted before getting feature importance."
        assert self.feature_names is not None, "Feature names must be provided before getting feature importance."
        # get quantity appropriate to the model type
        feature_importance = self.model.feature_importances_ if self.model_type == "xgb" else self.model.coef_[0]
        print('Feature importance:' if self.model_type == 'xgb' else 'Model coefficients:')
        for name, importance in zip(self.feature_names, feature_importance):
            print(f"{name}: {importance:0.2f}")
