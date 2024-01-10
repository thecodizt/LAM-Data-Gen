import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf

class GAN:
    def __init__(self, dataframe, randomness_degree):
        self.dataframe = dataframe
        self.randomness_degree = randomness_degree
        self.min_val = self.dataframe.min()
        self.max_val = self.dataframe.max()
        self.dataframe = self.normalize(self.dataframe)
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.gan = self.build_gan()

    def build_generator(self):
        model = Sequential()
        model.add(Dense(units=256, input_dim=self.randomness_degree))
        model.add(tf.keras.layers.LeakyReLU(0.2))
        model.add(Dense(units=512))
        model.add(tf.keras.layers.LeakyReLU(0.2))
        model.add(Dense(units=1024))
        model.add(tf.keras.layers.LeakyReLU(0.2))
        model.add(Dense(units=self.dataframe.shape[1], activation='tanh'))  # Changed from 'sigmoid' to 'tanh'
        model.compile(loss='binary_crossentropy', optimizer='adam')
        return model

    def build_discriminator(self):
        model = Sequential()
        model.add(Dense(units=1024, input_dim=self.dataframe.shape[1]))
        model.add(tf.keras.layers.LeakyReLU(0.2))
        model.add(Dense(units=512))
        model.add(tf.keras.layers.LeakyReLU(0.2))
        model.add(Dense(units=256))
        model.add(tf.keras.layers.LeakyReLU(0.2))
        model.add(Dense(units=1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam')
        return model

    def build_gan(self):
        self.discriminator.trainable = False
        gan_input = tf.keras.Input(shape=(self.randomness_degree,))
        x = self.generator(gan_input)
        gan_output= self.discriminator(x)
        gan= tf.keras.Model(inputs=gan_input, outputs=gan_output)
        gan.compile(loss='binary_crossentropy', optimizer='adam')
        return gan

    def normalize(self, data):
        return (data - self.min_val) / (self.max_val - self.min_val)

    def denormalize(self, normalized_data):
        denormalized_data = normalized_data * (self.max_val.values - self.min_val.values) + self.min_val.values
        return denormalized_data



    def generate(self, num_samples):
        noise = np.random.normal(0, 1, (num_samples, self.randomness_degree))
        generated_data = self.generator.predict(noise)
        generated_data = generated_data / 2 + 0.5  # Rescale from [-1, 1] to [0, 1]
        return self.denormalize(generated_data)