import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets


class Dataset:
    def __init__(self):
        full_data = datasets.load_iris()
        self.data = np.concatenate((full_data["data"], full_data["target"].reshape(-1, 1)), axis=1)
        self.feature_names = full_data["feature_names"]
        self.target_names = {i: name for i, name in enumerate(full_data["target_names"])}

    def get_train_val_data_with_labels(self, val_size: float = 0.2, random_state: int = 42) -> (
            np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        """Split the dataset into train and test and return the train and test data and labels.
        :param val_size: The size of the validation set.
        :param random_state: The random state to use for the train/test split.
        :return: The train and test features and labels."""
        # split the data into train and test
        train_data, val_data = train_test_split(self.data, test_size=val_size, random_state=random_state)
        return train_data[:, :-1], train_data[:, -1], val_data[:, :-1], val_data[:, -1]

    def get_all_data_with_labels(self) -> (np.ndarray, np.ndarray):
        """Return the whole dataset split into features and labels.
        :return: The features and labels."""
        return self.data[:, :-1], self.data[:, -1]

    def __len__(self) -> int:
        """Return the number of examples in the dataset"""
        return self.data.shape[0]
