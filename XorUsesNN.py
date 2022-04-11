from NN import *
import matplotlib.pyplot as plt

# https://www.geeksforgeeks.org/implementation-of-artificial-neural-network-for-xor-logic-gate-with-2-bit-binary-input/


def sigmoid(z):
    z = 1 / (1 + np.exp(-z))
    return z


def main():
    print("XorUsesNN.py")
    np.random.seed(2)
    layer1 = Layer(2, 2, sigmoid)
    layer2 = Layer(2, 1, sigmoid)
    nn = NN(verbose=True)
    nn.add_layer(layer1)
    nn.add_layer(layer2)
    # Inputs
    x = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])

    # These are XOR outputs
    y = np.array([[0, 1, 1, 0]])

    # These are OR outputs
    # y = np.array([[0, 1, 1, 1]])

    # These are And outputs
    # y = np.array([[0, 0, 0, 1]])

    nn.train(x, y)

    # We plot losses to see how our network is doing
    plt.plot(nn.losses)
    plt.xlabel("EPOCHS")
    plt.ylabel("Loss value")
    plt.show()

    test_values = nn.test(x)
    for i in range(x.shape[1]):
        print(f"{x[:, i]}, {test_values[0 , i]} {test_values[0 , i] >= 0.5}")


main()
