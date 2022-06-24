import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, \
    f1_score

from data import Dataset
from model import Model
from utils import load_config


def train(train_X: np.ndarray, train_y: np.ndarray, model_type: str = "xgb",
          model_path: str = "model_directory/model.pkl", feature_names: list = (),
          target_names: dict = dict()) -> Model:
    """Train and save the model.
    :param train_X: The features to train on.
    :param train_y: The labels to train on.
    :param model_type: The type of model to train.
    :param model_path: The path to save the model to.
    :param feature_names: The names of the features.
    :param target_names: The target names dictionary.
    :return: The trained model.
    """
    model = Model(model_type=model_type)
    model.set_feature_names(feature_names)
    model.set_target_names(target_names)
    model.train(train_X, train_y)
    model.save(model_path)
    return model


def evaluate(model: Model, X: np.ndarray, y_actual: np.ndarray):
    """Evaluate the model on the given data using mean squared error, r squared, mean absolute error,
     and root mean squared error.

    :param model: The model to evaluate.
    :param X: The features to evaluate on.
    :param y_actual: The actual labels to evaluate on.
     """
    y_pred = model.predict(X)
    print(f"Accuracy: {accuracy_score(y_actual, y_pred):.2f}")
    print(f"Confusion matrix: \n{confusion_matrix(y_actual, y_pred)}")
    print(f"Precision: {precision_score(y_actual, y_pred, average='micro'):.2f}")
    print(f"Recall: {recall_score(y_actual, y_pred, average='micro'):.2f}")
    print(f"F1 score: {f1_score(y_actual, y_pred, average='micro'):.2f}")


def train_full_model(config: dict) -> Model:
    """Train the model on the full dataset.
    :param config: The configuration dictionary.
    :return: The trained model.
    """
    dataset = Dataset()
    X, y = dataset.get_all_data_with_labels()
    model = train(X, y, model_type=config['model']['type'],
                  model_path=config['model']['path'], feature_names=dataset.feature_names,
                  target_names=dataset.target_names)
    return model


def main():
    """Train and evaluate the model."""
    # load config
    config = load_config()
    np.random.seed(config['training']['random_seed'])
    # getting the data
    dataset = Dataset()
    train_X, train_y, val_X, val_y = dataset.get_train_val_data_with_labels(
        val_size=config["training"]["validation_size"], random_state=config["training"]["random_seed"])
    # training the model
    model = train(train_X, train_y, model_type=config['model']['type'],
                  model_path=config['model']['path'], feature_names=dataset.feature_names)
    model.print_features_importance()
    print()

    # evaluating performance on training and validation sets
    split = ["train", "validat"]
    data = [(train_X, train_y), (val_X, val_y)]
    for s, data_split in zip(split, data):
        print(f"Evaluating {s}ing performance")
        evaluate(model, data_split[0], data_split[1])
        print("\n")
    # retrain for the whole dataset
    print("Training on the full dataset.")
    final_model = train_full_model(config)
    return final_model


if __name__ == "__main__":
    main()
