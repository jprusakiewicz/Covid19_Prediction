import sys
from omegaconf import OmegaConf
from sklearn.pipeline import make_pipeline

sys.path.append('src')

from data.read_data import read_data
from data.preprocess import preprocess_data, get_preprocessor
from models.sklearn import build_model as build_sklearn_model
from models.kerass import build_model as build_keras_model
from evaluate import evaluate_model


# todo kuba save artifacts (config, model, metrics, etc.) in one place
def run() -> dict:
    config = OmegaConf.load('config/test_config.yaml')
    data = read_data()
    x_train, x_test, y_train, y_test = preprocess_data(data)

    match config.model.model_library:
        case "keras":
            pipeline = build_keras_model(x_train, y_train, config.model)

        case "sklearn":
            pipeline = make_pipeline(get_preprocessor(config.preprocessing),
                                     build_sklearn_model(config.model), verbose=2)
            pipeline.fit(x_train, y_train)

        case _:
            raise ValueError(f"unsupported model library: {config.model.model_library}")

    metrics = evaluate_model(model=pipeline, x=x_test, y=y_test)
    return metrics


if __name__ == "__main__":
    metrics = run()
    print(metrics)
