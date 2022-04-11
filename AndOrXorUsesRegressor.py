import numpy as np
from Regressor import Regressor


def sigmoid(z):
    rv = 1/(1 + np.exp(-z))
    return rv


def cross_entropy(Y_hat, Y):
    rv = -np.sum(Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat))
    return rv


def main():
    x_training_data = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
    # OR outputs:
    # y_training_data = np.array([[0, 1, 1, 1]])

    # AND outputs:
    # y_training_data = np.array([[0, 0, 0, 1]])

    # XOR outputs
    y_training_data = np.array([[0, 1, 1, 0]])

    # Create the model
    model = Regressor(sigmoid, cross_entropy, normalized=False)

    # Train the model and create predictions
    model.fit(x_training_data, y_training_data)
    for x_test_data in x_training_data.T:
        predictions = model.predict(x_test_data) >= 0.5
        print(f"x_test={x_test_data.T} ,prediction={predictions}")


if __name__ == '__main__':
    main()
