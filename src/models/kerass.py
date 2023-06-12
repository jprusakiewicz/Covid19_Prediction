from tensorflow import keras

LAYERS_MAPPING = {
    "SimpleRNN": keras.layers.SimpleRNN,
    "LSTM": keras.layers.LSTM,
    "Dense": keras.layers.Dense,
}


def build_model(x_train, y_train, config):
    keras.utils.set_random_seed(config.random_state)

    model = keras.Sequential()
    for layer in config.layers:
        layer_instance = LAYERS_MAPPING[layer.type](**layer.params)
        if 'bidirectional' in layer and layer.bidirectional:
            layer_instance = keras.layers.Bidirectional(layer_instance)
        model.add(layer_instance)

    early_stopping = keras.callbacks.EarlyStopping(monitor=config.early_stopping.monitor,
                                                   patience=config.early_stopping.patience,
                                                   mode=config.early_stopping.mode)

    model.compile(loss=keras.losses.MeanSquaredError(),
                  optimizer=keras.optimizers.Adam(),
                  metrics=[keras.metrics.MeanAbsoluteError()])

    model.fit(x_train,
              y_train,
              epochs=config.epochs,
              callbacks=[early_stopping],
              verbose=3)

    return model
