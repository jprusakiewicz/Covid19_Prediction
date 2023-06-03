import tensorflow as tf
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

    model = tf.keras.Sequential([
        keras.layers.LSTM(32),
        tf.keras.layers.Dense(units=16, activation="relu"),
        tf.keras.layers.Dense(units=1)

    ])
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=6,
                                                      mode='min')

    model.compile(loss=tf.keras.losses.MeanSquaredError(),
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=[tf.keras.metrics.MeanAbsoluteError()])

    model.fit(dataset_train,
              epochs=40,
              batch_size=30,
              # validation_data=dataset_val,
              callbacks=[early_stopping])

    return model
