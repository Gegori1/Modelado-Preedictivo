import tensorflow as tf

class NNRegressor(tf.keras.Model):
    def __init__(self, input_shape: int=200, activation: str="relu"):
        super().__init__()
        self.dense_1 = tf.keras.layers.Dense(input_shape, activation=activation)
        # self.dense_2 = tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))
        # self.dense_3 = tf.keras.layers.Dense(8, activation='sigmoid')
        self.dense_4 = tf.keras.layers.Dense(1, activation='linear')

    def call(self, inputs):
        x = self.dense_1(inputs)
        # x = self.dense_2(x)
        # x = self.dense_3(x)
        return self.dense_4(x)

    def compile_model(self, optimizer: str='adam', loss: str='mse', metrics: list=['mse']):
        loss_fn = loss
        optimizer = optimizer
        self.compile(loss=loss_fn, optimizer=optimizer, metrics=metrics)


if __name__ == '__main__':
    model = NNRegressor()
    model.compile_model()
    model.build(input_shape=(None, 24))
    model.summary()
    # tf.keras.utils.plot_model(model, 'my_first_model.png', show_shapes=True)