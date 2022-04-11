import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class Regressor:
    def __init__(self, hypothesis, cost, normalized=True, alpha=0.1, epochs=500000, verbose=True):
        self.hypothesis = hypothesis
        self.cost = cost
        self.sc = StandardScaler()
        self.W = None
        self.b = 0
        self.normalized = normalized
        self.alpha = alpha
        self.epochs = epochs
        self.verbose = verbose
        self.fitted = False
        self.X_in = None
        self.y_in = None

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if isinstance(y, pd.Series):
            y = y.to_numpy().reshape(1, -1)
        count = y.shape[1]

        self.X_in = X
        self.y_in = y
        if self.normalized:
            X = self.sc.fit_transform(X.T).T

        prev_cost = None
        self.W = np.zeros(X.shape[0]).reshape(1, -1)
        # self.W = np.random.rand(1, X.shape[0])
        self.b = 0

        for i in range(self.epochs):
            z = self.W @ X + self.b
            y_hat = self.hypothesis(z)
            dw = (1 / count) * X @ (y_hat - y).T
            db = (1 / count) * np.sum((y_hat - y))
            self.W -= self.alpha * dw.T
            self.b -= self.alpha * db.T

            cost = self.cost(y_hat, y)
            if i % 1000 == 0 or i < 10:
                if self.verbose:
                    print(f"Regressor.fit(): cost after {i} iterations is {cost}")

            if prev_cost is not None and np.abs(prev_cost - cost) < 1e-15:
                if self.verbose:
                    print(f"Regressor.fit(): Converged at iteration {i}, cost= {cost}")
                break
            prev_cost = cost

        print(f"Regressor.fit(): fitted  values W={self.W} b={self.b}")
        self.fitted = True

    def predict(self, X):
        if not self.fitted:
            raise ValueError("Can not predict unfitted model, please call fit() before calling predict().")
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        if self.normalized:
            X = self.sc.transform(X.T).T

        z = self.W @ X + self.b
        y_hat = self.hypothesis(z)
        return y_hat

    def show(self):
        if not self.fitted:
            raise ValueError("Can not predict unfitted model, please call fit() before calling predict().")
        if len(self.W[0]) != 2:
            raise ValueError(f"Can show only model with 2 dimensions. Current model has {len(self.W)} dimensions.")

        # Calculate the intercept and gradient of the decision boundary.
        if self.normalized:
            raise ValueError("TBD")
            b = self.b * self.sc.scale_[0] * self.sc.scale_[1] + self.sc.mean_[0] + self.sc.mean_[1]
            W = self.W * self.sc.scale_ + self.sc.mean_
            c = -b / W[0][1]
            m = -self.W[0][0] / self.W[0][1]
            c = 16000

            """
            m =  -self.W[0][0] / self.W[0][1] * self.sc.scale_[0] / self.sc.scale_[1]
            c = - self.sc.scale_[1] * self.sc.scale_[0] * self.b / self.W[0][1] + self.W[0][0] * self.sc.mean_[0] + \
                self.W[0][1] * self.sc.mean_[1]
            """

        else:

            c = -self.b / self.W[0][1]
            m = -self.W[0][0] / self.W[0][1]

        print(f"b= {self.b} ,w1= {self.W[0][0]} ,w2= {self.W[0][1]}")
        print(f"m= {m} , c= {c}")
        x_min = min(self.X_in[0])
        x_max = max(self.X_in[0])
        y_min = min(self.X_in[1])
        y_max = max(self.X_in[1])
        xd = np.array([x_min, x_max])
        yd = m * xd + c

        print(f"xd= {xd}")
        print(f"yd= {yd}")

        plt.plot(xd, yd, 'k', lw=1, ls='--')
        plt.fill_between(xd, yd, y_min, color='tab:blue', alpha=0.2)
        plt.fill_between(xd, yd, y_max, color='tab:orange', alpha=0.2)

        plt.scatter(*self.X_in[:, self.y_in[0] == 0], s=38, alpha=0.5)
        plt.scatter(*self.X_in[:, self.y_in[0] == 1], s=38, alpha=0.5)
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.ylabel(r'$x_2$')
        plt.xlabel(r'$x_1$')

        plt.show()
