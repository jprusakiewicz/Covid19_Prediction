import json
import itertools
import inspect

from omegaconf import OmegaConf


def generate_sklearn_params():
    models = [
        {"model_type": "DecisionTreeRegressor",
         "params": {"criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"],
                    "min_samples_leaf": range(1, 10, 1), "ccp_alpha": range(1, 5, 1)}},

        {"model_type": "BaggingRegressor",
         "params": {"n_jobs": [-1], "n_estimators": range(5, 25, 5), "max_samples": range(1, 5, 1)}},

        {"model_type": "RandomForestRegressor",
         "params": {"criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"],
                    "n_estimators": range(50, 250, 50), "min_samples_split": range(1, 5, 1)}},

        {"model_type": "KernelRidge",
         "params": {"alpha": [0.1, 1.0, 3.0, 10.0], "gamma": [0.001, 0.01, 0.1, 1.0], "kernel": ["linear", "rbf"]}},

        {"model_type": "GaussianProcessRegressor",
         "params": {"alpha": [0.1, 1.0, 3.0, 10.0], "n_restarts_optimizer": [0, 1, 2]}},

        {"model_type": "GradientBoostingRegressor",
         "params": {'learning_rate': [0.1, 0.01],
                    "subsample": [0.01, 0.1, 0.5, 1.0],
                    'n_estimators': range(50, 250, 50),
                    'max_depth': range(1, 5, 1),
                    'min_samples_split': range(1, 5, 1)}},

        {"model_type": "MLPRegressor",
         "params": {'hidden_layer_sizes': [(10,), (20,), (10, 10)],
                    'activation': ['relu', 'tanh'],
                    'alpha': [0.0001, 0.001, 0.01],
                    'learning_rate': ['constant', 'adaptive']}}
    ]

    combinations = []
    for model in models:
        for items in itertools.product(*model["params"].values()):
            combination_dict = {"params": {var_name: item for var_name, item in zip(model["params"].keys(), items)}}
            combination_dict.update({"model_type": model["model_type"]})
            if model["model_type"] in ["DecisionTreeRegressor", "BaggingRegressor", "RandomForestRegressor",
                                       "GaussianProcessRegressor", "GradientBoostingRegressor", "MLPRegressor"]:
                combination_dict["params"].update({"random_state": 420})
            combinations.append(combination_dict)

    return combinations


def bind_sklearn_config(config_params):
    model = {"model_library": "sklearn", "type": config_params['model_type'], "params": config_params['params']}

    preprocessing = {"additional_features": False,
                     "sequence_length": 27,
                     "use_scaler": True,
                     "train_test_split":
                         {
                             "test_size": 0.3,  # do not change
                             "random_state": 420
                         }
                     }
    random_state: 420
    config = {"model": model, "preprocessing": preprocessing, "mlflow": {"tracking_uri": "http://localhost:5000"}}
    conf = OmegaConf.create(config)
    config_as_json = OmegaConf.to_container(conf)
    return config_as_json
    # with open("config/generated_configs/test_sk_config.yaml", 'wt') as config_file:
    #     OmegaConf.save(config=conf, f=config_file)


def generate_keras_params():
    layer_type = ["LSTM", "SimpleRNN"]
    sequence_length = range(3, 29, 4)
    units_first_layer = range(8, 64, 16)
    epochs = range(40, 240, 40)
    additional_features = [True, False]
    bidirectional = [True, False]
    use_scaler = [True, False]

    params = [layer_type, sequence_length, units_first_layer, epochs, additional_features, bidirectional, use_scaler]
    # Get the variable names dynamically
    variable_names = [var_name for var_name, var_value in inspect.currentframe().f_locals.items() if
                      var_value in params]

    # Generate all combinations with variable names
    combinations = []
    for items in itertools.product(*params):
        combination_dict = {var_name: item for var_name, item in zip(variable_names, items)}
        combinations.append(combination_dict)

    return combinations


def bind_keras_config(sequence_length, additional_features, use_scaler, bidirectional, epochs, layer_type,
                      units_first_layer):
    model_library = "keras"
    early_stopping = {"monitor": "mean_absolute_error",  # try also "loss"
                      "patience": 5,
                      "mode": "min"}

    layers = [
        {"type": layer_type, "bidirectional": bidirectional,
         "params": {"units": units_first_layer, "activation": "relu"}},
        {"type": "Dense", "params": {"units": 8, "activation": "relu"}},
        {"type": "Dense", "params": {"units": 4, "activation": "relu"}},
        {"type": "Dense", "params": {"units": 2, "activation": "relu"}},
        {"type": "Dense", "params": {"units": 1}},

    ]
    model = {"model_library": model_library, "epochs": epochs, "random_state": 420,
             "layers": layers, "early_stopping": early_stopping}

    preprocessing = {"additional_features": additional_features,
                     "sequence_length": sequence_length,
                     "use_scaler": use_scaler,
                     "train_test_split":
                         {
                             "test_size": 0.3,  # do not change
                             "random_state": 420
                         }
                     }
    config = {"model": model, "preprocessing": preprocessing, "mlflow": {"tracking_uri": "http://localhost:5000"}}
    conf = OmegaConf.create(config)
    config_as_json = OmegaConf.to_container(conf)
    return config_as_json
    # config_hash = hash(conf)
    # with open(f"config/generated_configs/test_keras_config_{config_hash}.yaml", 'w') as config_file:
    #     print(config_hash)
    #     OmegaConf.save(config=conf, f=config_file)


def generate_sklearn_configs():
    global params, f
    sklearn_params = generate_sklearn_params()
    sklearn_configs = [bind_sklearn_config(params) for params in sklearn_params]
    with open(f"config/generated_configs/config_sklearn_additional_no_features_with_scaler.json", 'w') as f:
        json.dump(sklearn_configs, f)


def generate_keras_configs():
    keras_params = generate_keras_params()
    keras_configs = [bind_keras_config(**params) for params in keras_params]
    print(len(keras_configs))
    with open(f"config/generated_configs/config_keras.json", 'w') as f:
        json.dump(keras_configs, f)


if __name__ == "__main__":
    # generate_keras_configs()
    generate_sklearn_configs()
