from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor


def build_model(config):
    match config.type:
        case "DecisionTreeRegressor":
            model = DecisionTreeRegressor
        case "BaggingRegressor":
            model = BaggingRegressor
        case "RandomForestRegressor":
            model = RandomForestRegressor
        case "KernelRidge":
            model = KernelRidge
        case "GaussianProcessRegressor":
            model = GaussianProcessRegressor
        case "GradientBoostingRegressor":
            model = GradientBoostingRegressor
        case "MLPRegressor":
            model = MLPRegressor
        case _:
            raise ValueError(f"model type {config.model.type} not supported")
    return model(**config.params)


def run_training(x, y, config):
    model = build_model(config.model)
    model.fit(x, y)
    return model
