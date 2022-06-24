import os

import flask
import numpy as np
from flask import request, jsonify, Response

from model import Model
from train import train_full_model
from utils import load_config


def main():
    """ Start the server app running."""
    # initialising the app
    app = flask.Flask(__name__)
    app.config["DEBUG"] = True
    config = load_config()
    # loading the model
    model_path = config['model']['path']
    if os.path.isfile(model_path):
        model = Model.load(None, path=model_path)
    # if there is no saved model, train the model
    else:
        model = train_full_model(config)

    @app.route('/api/get_iris_class', methods=['GET'])
    def predict_woz_value() -> Response:
        """ Predict the iris class value for a given set of features."""
        # convert request to np.ndarray with dimensionality (1, num_features)
        x = np.fromiter(map(int, request.args.values()), dtype=int).reshape(1, len(model.feature_names))
        # get the prediction
        iris_class = model.predict(x)
        # convert to name
        class_name = model.target_names[int(iris_class)]
        # convert to json
        iris_jason = jsonify({"iris_class": class_name})
        return iris_jason

    app.run(port=8000, host="0.0.0.0")


if __name__ == "__main__":
    main()
