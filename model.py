import numpy as np
import game
import tensorflow as tf


INPUT_SIZE = len(game.AbstractCar.angles_ray) + 1
OUTPUT_SIZE = len(game.AbstractCar.ai_possible_moves)


class CarNet:
    bias_init = tf.keras.initializers.RandomUniform(minval=-0.01, maxval=0.01, seed=None)

    def __init__(self, population_size, perceptron_by_car=128):
        self.pbc = perceptron_by_car
        self.population_size = population_size
        self.model = self.get_full_model()

    def get_full_model(self):
        models = [self.get_sub_model() for _ in range(self.population_size)]
        model = tf.keras.models.Model(inputs=[m.input for m in models], outputs=[m.output for m in models])
        return model

    def get_sub_model(self):
        inp = tf.keras.layers.Input(shape=(INPUT_SIZE,))
        x = tf.keras.layers.Dense(self.pbc, activation="relu", kernel_initializer='random_normal', bias_initializer=CarNet.bias_init)(inp)
        x = tf.keras.layers.Dense(OUTPUT_SIZE, activation="softmax", kernel_initializer='random_normal', bias_initializer=CarNet.bias_init)(x)
        _model = tf.keras.models.Model(inputs=inp, outputs=x)
        return _model

    def get_weights_and_biases(self):
        weights, biases = [], []
        for layer in self.model.layers:
            if not layer.get_weights():
                continue
            w = layer.get_weights()[0]
            b = layer.get_weights()[1]
            weights.append(w)
            biases.append(b)
        weights, biases = np.array(weights, dtype="object"), np.array(biases, dtype="object")
        return [weights[:self.population_size], weights[self.population_size:]], \
               [biases[:self.population_size], biases[self.population_size:]]

    def set_weights_and_biases(self, weights, biases):
        self.model.set_weights([x for t in zip(weights, biases) for x in t])

    @tf.function(jit_compile=False)
    def make_prediction(self, X):
        with tf.device('/gpu:0'):
            return self.model(X, training=False)

    def save(self):
        tf.saved_model.save(self.model, f"data/models/{self.population_size}/unlimited")
