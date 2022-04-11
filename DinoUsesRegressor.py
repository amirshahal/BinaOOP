import numpy as np
from Regressor import Regressor


def sigmoid(z):
    z = np.clip(z, -700, 700)
    rv = 1/(1 + np.exp(-z))
    return rv


def cross_entropy(Y_hat, Y):
    rv = -np.sum(Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat))
    return rv


def main():
    # Dino's input
    x_training_data = np.array([
        [10000, 9500, 9000, 9300, 9700, 9300, 9700, 9100, 8500, 8800, 8600, 8700, 9000, 8300, 9100, 8900],
        [12000, 11000, 10550, 10300, 12500, 12300, 11500, 11500, 10000, 8800, 8600, 9500, 9000, 10000, 8500, 8300]])

    x_training_data = x_training_data / 10000

    # Dino's output
    y_training_data = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]).reshape(1, -1)

    use_sklearn = False
    if use_sklearn:
        import sklearn.linear_model
        model = sklearn.linear_model.LogisticRegression()
        model.fit(x_training_data.T, y_training_data.ravel())
        print(model.coef_, model.intercept_)
        for i, x_test_data in enumerate(x_training_data.T):
            predictions = model.predict(x_test_data.reshape(1, -1))
            s = "   <<<---" if predictions[0] != y_training_data[0][i] else ""
            print(f"{i+1} x_test={x_test_data.T} ,prediction={predictions},  y={y_training_data[0][i]} {s}")
        import sys
        sys.exit()

    # Create the model
    model = Regressor(sigmoid, cross_entropy, normalized=False, alpha=10, epochs=5000)

    # Train the model and create predictions
    model.fit(x_training_data, y_training_data)

    for i, x_test_data in enumerate(x_training_data.T):
        predictions = model.predict(x_test_data.reshape(2, 1)) >= 0.5
        s = "   <<<---" if predictions[0] != y_training_data[0][i] else ""
        print(f"{i} x_test={x_test_data.T} ,prediction={predictions[0]} , y={y_training_data[0][i]} {s}")

    model.show()


if __name__ == '__main__':
    main()
