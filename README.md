Simple project which trains different models on the iris dataset. The models are either logistic regression or xgboost.
The model type (as well as model path and seed) is specified in the `config.yaml` file.

The code is in the `src` folder.
The model is trained and evaluated in the `train.py` file. It is the "main" file of the project and its `main` function
trains and evaluates the model. The `data.py` file contains the data preprocessing, the `model.py` file contains the
model API and the `utils.py` file contains the helper function.

To install this as a package, run `pip install -e .`

The model is wrapped with a simple flask server in the `app.py` file.

The GET API is used to get the model predictions of the class and is as follows:

    http://0.0.0.0:8000/api/get_iris_class?f1=v1&f2=v2&f3=v3&f4=v4

where `f1...4` are the features of the iris dataset and `v1...4` are the values.

A docker file is provided to wrap around the package and run the flask server. Run the following
command to build the image, start the image and start the flask server:

    docker build -t iris_flask .
    docker run -p 8000:8000 -t iris_flask

To test the flask server, run the following command:

    curl "http://0.0.0.0:8000/api/get_iris_class?f1=1&f2=2&f3=3&f4=4"
