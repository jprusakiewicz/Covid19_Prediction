from tensorflow import keras


def WIP_keras_model(x_train, y_train):
    SEQUENCE_LENGTH = 3
    dataset_train = keras.preprocessing.timeseries_dataset_from_array(
        x_train,
        y_train,
        sequence_length=SEQUENCE_LENGTH
    )

    # dataset_val = keras.preprocessing.timeseries_dataset_from_array(
    #     x_val,
    #     y_val,
    #     sequence_length=14,
    #     #     sampling_rate=step,
    #     #     batch_size=batch_size,
    # )

    for batch in dataset_train.take(1):
        inputs, targets = batch

    print("Input shape:", inputs.numpy().shape)
    print("Input value:", inputs.numpy()[0][0][0])
    print("Input value:", inputs.numpy()[0][1][0])
    # print("Input value:", inputs.numpy()[0][2][0])

    print("Target shape:", targets.numpy().shape)
    print("Target value:", targets.numpy()[0])

    keras.utils.set_random_seed(420)  # todo config

    # model = tf.keras.Sequential([
    #     keras.layers.LSTM(32),
    #     keras.layers.Dense(units=4, activation="relu"),
    #     keras.layers.Dense(units=1)
    # ])

    # model = keras.Sequential([
    #     keras.layers.SimpleRNN(units=16, input_shape=(1, 1), activation="relu"),
    #     keras.layers.Dense(units=4, activation="relu"),
    #     keras.layers.Dense(units=1)
    # ])
    model = keras.Sequential([
        keras.layers.Bidirectional(
        keras.layers.SimpleRNN(units=8, input_shape=(1, 1), activation="relu"),
        ),
        keras.layers.Dense(units=8, activation="relu"),
        keras.layers.Dense(units=4, activation="relu"),
        keras.layers.Dense(units=1)

    ])

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=6,
                                                      mode='min')

    model.compile(loss=keras.losses.MeanSquaredError(),
                  optimizer=keras.optimizers.Adam(),
                  metrics=[keras.metrics.MeanAbsoluteError()])

    model.fit(dataset_train,
              epochs=40,
              batch_size=35,
              # validation_data=dataset_val,
              callbacks=[early_stopping])

    return model
