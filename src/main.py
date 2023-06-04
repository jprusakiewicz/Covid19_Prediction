import sys
from omegaconf import OmegaConf
from sklearn.pipeline import make_pipeline

sys.path.append('src')

from read_data import read_data
from train import build_model
from evaluate import evaluate_model
from preprocess import preprocess_data, get_preprocessor
from kerass import build_keras_model


# todo kuba save artifacts (config, model, metrics, etc.) in one place
def run() -> dict:
    config = OmegaConf.load('config/test_config.yaml')
    data = read_data()
    x_train, x_test, y_train, y_test = preprocess_data(data)

    if config.model.model_library == "keras":
        pipeline = build_keras_model(x_train, y_train, config.model)
    elif config.model.model_library == "sklearn":
        pipeline = make_pipeline(get_preprocessor(config.preprocessing),
                                 build_model(config.model), verbose=2)
        pipeline.fit(x_train, y_train)

    else:
        raise ValueError(f"unsupported model library: {config.model.model_library}")

    metrics = evaluate_model(model=pipeline, x=x_test, y=y_test)
    return metrics


if __name__ == "__main__":
    metrics = run()
    print(metrics)
