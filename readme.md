# Timeseries regression prediction
## Template project for training and evaluating sklearn and keras based models

### How to use (tl;dr)
1. Download data using `make download-bash` command (currently only bash env supported)
2. create/activate python venv with `python -m venv venv` and `source venv/bin/bash` commands **python >3.10 required!**
3. install requirements with `pip install -r requirements.txt` command
4. run mlflow with `make run_mlflow` command (it will run on __localhost:5000__ by default)
5. run main.py with `python src/main.py` command

## Running experiments
### configs 
In order to run experiments (using `main.py` file) you need to specify configuration in `.json` or `.yaml` file. They are all stored in `config` directory.  
Note that `json` format was introduced to this project in order to run multiple experiments at once. If you want to run single experiment, you should choose `yaml` format for simplicity.

### yaml config structure
#### keras
Currently available keras architectures:
- SimpleRNN
- LSTM

```yaml
mlflow:
  tracking_uri: "http://localhost:5000" # url that mlflow was run on

preprocessing:
  additional_features: True # if true additional features will be mined
  sequence_length: 27 #  number of days to predict (timeseries sliding window size)
  use_scaler: False # if true numerical features will be scaled (don't worry, there's no data leak, test and train data scaled separately).
  train_test_split:
    test_size: 0.3
    random_state: 420

model:
  model_library: "keras" # what library to use, currently supported values:  "keras" and "sklearn"
  random_state: 420
  epochs: 40
  layers: # keras layers list, currently supported values:  "SimpleRNN" and "LSTM" and "Dense". You can remove or add more/fewer layers here.
    - type: "SimpleRNN"
      bidirectional: true
      params:
        units: 16
        activation: "relu"
    - type: "Dense"
      params:
        units: 8
        activation: "relu"
    - type: "Dense"
      params:
        units: 4
        activation: "relu"
    - type: "Dense"
      params:
        units: 1
  early_stopping:
    monitor: "mean_absolute_error" # you can try also "loss"
    patience: 5
    mode: "min"
```
#### sklearn
Currently available sklearn models:
- DecisionTreeRegressor
- BaggingRegressor
- RandomForestRegressor
- KernelRidge
- GaussianProcessRegressor
- GradientBoostingRegressor
- MLPRegressor

```yaml
mlflow:
  tracking_uri: "http://localhost:5000"

preprocessing:
  additional_features: True
  sequence_length: 27
  use_scaler: False
  train_test_split:
    test_size: 0.3
    random_state: 420

model:
  model_library: "sklearn"
  type: "GradientBoostingRegressor"
  params: # all the params that can be passed as kwargs to models constructor (like this: GradientBoostingRegressor(**params))
    subsample: 0.5
    learning_rate: 0.01
    random_state: 420
```

### Ealuation metrics
`RMSE` and `R2` are metrics that were used. You can change them in `src/evaluate.py` file.

### Generate new configs for gridsearch
use `config/generate_configs.py` for this case
