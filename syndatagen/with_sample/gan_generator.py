import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import os

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
    
def compare_data(original_data, generated_data, generated_name):
    # Calculate mean and standard deviation for each column
    original_stats = original_data.describe().loc[['mean', 'std']]
    generated_stats = generated_data.describe().loc[['mean', 'std']]

    print("Original Data Statistics:")
    print(original_stats)
    
    print("\nGenerated Data Statistics:")
    print(generated_stats)

    save_folder = 'syndatagen/with_sample/generated_data/statistics'
    os.makedirs(save_folder, exist_ok=True)

    # Save both original and generated statistics to a CSV file
    stats_filename = os.path.join(save_folder, f'{generated_name}_combined_statistics.csv')
    combined_stats = pd.concat([original_stats, generated_stats])
    combined_stats.to_csv(stats_filename)

    # Plot a comparison chart and save it
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original Data")
    original_data.boxplot()
    
    plt.subplot(1, 2, 2)
    plt.title("Generated Data")
    generated_data.boxplot()

    plt.savefig(os.path.join(save_folder, f'{generated_name}_comparison.png'))
    plt.show()

def filter_numeric_columns(data):
        #temporary function to extract only numeric columns
        return data.select_dtypes(include=[np.number])

def main():
    #function to temporarily test the gan.
    # takes a csv input (here takes data/heart.csv)
    #  will store the generated df inside generated/data, statistics and images will be stored in generated/stats
    path  = 'data/heart.csv'
    num_generated_samples = 1000

    csv_file_name = os.path.splitext(os.path.basename(path))[0]
    generated_name = f'generated_{csv_file_name}_{num_generated_samples}'
    dataframe = pd.read_csv(path)
    numeric_dataframe = filter_numeric_columns(dataframe)

    #testing
    randomness_degree = 100
    gan_model = GAN(numeric_dataframe, randomness_degree)

    generated_samples = pd.DataFrame(gan_model.generate(num_generated_samples))
    generated_samples.to_csv('syndatagen/with_sample/generated_data/dataframe/'+generated_name, index=False)
    compare_data(numeric_dataframe, generated_samples, generated_name)
    
if __name__ == "__main__":
    main()