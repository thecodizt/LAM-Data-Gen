# Data Generator - 2
## Supported Methods

**With Sample Data:**

1. **Deep Learning Techniques:**
    - **Generative Adversarial Networks (GANs):** GANs consist of two neural networks, a generator and a discriminator, that are trained simultaneously. The generator creates synthetic data, and the discriminator evaluates the data. The generator improves over time to produce data that the discriminator cannot distinguish from real data.
    - **Variational Autoencoders (VAEs):** VAEs are a type of autoencoder, a neural network used for data encoding. VAEs generate new data by learning the probability distribution of the input data, then sampling from this distribution.
    - **Recurrent Neural Networks (RNNs):** RNNs are a type of neural network designed to recognize patterns in sequences of data. They can be used to generate new sequences that mimic the patterns in the input data.

2. **Decision Trees:** Decision trees are a type of machine learning model that makes decisions based on the values of input features. For synthetic data generation, decision trees can be used in the following ways:
    - **Classification Trees:** These are used when the target variable is categorical. The tree is built by splitting the data based on the values of the input features that result in the largest decrease in class impurity.
    - **Regression Trees:** These are used when the target variable is continuous. The tree is built by splitting the data based on the values of the input features that result in the largest decrease in variance.

**Without Sample Data:**

1. **Iterative Proportional Fitting (IPF):** IPF is a technique for adjusting the cells of an initial table to match known marginal totals. You start with an initial guess for the joint distribution of the variables, and then iteratively adjust this distribution to match the known marginal distributions until convergence.

2. **Monte Carlo Method:** The Monte Carlo method uses repeated random sampling to estimate statistical properties. This method can be used to generate synthetic data based on known properties (like mean, variance, etc.) of the real data. You can generate random values for each feature in your dataset using the known distribution of the feature.

## Development Setup

1. Clone the repository

2. Open the cloned repository in Visual Studio Code.

3. Start the container using Docker Compose:

    - Make sure you have Docker installed on your machine.
    - Open a terminal in Visual Studio Code.
    - Run the following command to start the container:

        ```bash
        docker-compose up --build -d
        ```

4. Connect to the container with Dev Container:
    - Press `Ctrl + Shift + P` (or `Cmd + Shift + P` on macOS) to open the Command Palette.
    - Type "Dev Containers: Attach to Running Container" and select it.
    - Choose the running container from the list.

5. Change the directory to the working directory:
    - Open a terminal in Visual Studio Code.
    - Run the following command to change the directory:
        ```bash
        cd /code
        ```
