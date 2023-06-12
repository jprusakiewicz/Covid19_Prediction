import sys
from omegaconf import OmegaConf
from sklearn.pipeline import make_pipeline
import mlflow
from urllib3.exceptions import NewConnectionError

sys.path.append('src')

from data.read_data import read_data
from data.preprocess import preprocess_data
from models.sklearn import build_model as build_sklearn_model
from models.kerass import build_model as build_keras_model
from evaluate import evaluate_model


def run() -> dict:
    config = OmegaConf.load('config/test_config.yaml')
    data = read_data()
    x_train_2d, x_test_2d, x_train_3d, x_test_3d, y_train, y_test = preprocess_data(data, config.preprocessing)

    try:
        mlflow.set_tracking_uri(config.mlflow.tracking_uri)
    except NewConnectionError as e:
        print("MLFLOW connection error. Is mlflow running?")
        raise e
    with mlflow.start_run(run_name=f"test_run"):
        config_as_json = OmegaConf.to_container(config)
        mlflow.log_dict(config_as_json, 'config.json')
        mlflow.set_tag('config_hash', hash(frozenset(config_as_json)))

        match config.model.model_library:
            case "keras":
                mlflow.tensorflow.autolog()
                model = build_keras_model(x_train_3d, y_train, config.model)
                x_test = x_test_3d

            case "sklearn":
                mlflow.sklearn.autolog()
                model = make_pipeline(build_sklearn_model(config.model))
                model.fit(x_train_2d, y_train)
                x_test = x_test_2d

            case _:
                raise ValueError(f"unsupported model library: {config.model.model_library}")

        metrics = evaluate_model(model=model, x=x_test, y=y_test)
        mlflow.log_metrics(metrics)

    return metrics


if __name__ == "__main__":
    metrics = run()
    print(metrics)
