import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model
# plt.rc('text', usetex=True)


# https://scipython.com/blog/plotting-the-decision-boundary-of-a-logistic-regression-model/

data_set = 4

if data_set == 1:
        pts = np.loadtxt('linpts.txt')
        X = pts[:,:2]
        Y = pts[:,2].astype('int')
        xmin, xmax = -1, 2
        ymin, ymax = -1, 2.5

elif data_set == 2:
        # Works fine
        X = np.array([
                [10000, 9500,  9000, 9300,
                 9700,  9300,  9700, 9100,
                 8500,  8800,  8600, 8700,
                 9000, 8300, 9100, 8300],
                [12000, 11000, 10550, 10300,
                 12500, 12300, 11500, 11500,
                 8500, 8800, 8600, 8700,
                 9000, 8300, 9100, 8300]])

        xmin = min(X[0])
        xmax = max(X[0])
        ymin = min(X[1])
        ymax = max(X[1])
        X = X.T

        # Dino's output
        Y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])


elif data_set == 3:
        # Fails
        X = np.array([
                [10000, 9500,  9000, 9300,
                 9700,  9300,  9700, 9100,
                 8500,  8800,  8600, 8700,
                 9000, 8300, 9100, 8900],
                [12000, 11000, 10550, 10300,
                 12500, 12300, 11500, 11500,
                 10500, 8800, 8600, 9500,
             9000, 10000, 8500, 8300]])

        xmin = min(X[0])
        xmax = max(X[0])
        ymin = min(X[1])
        ymax = max(X[1])
        X = X.T

        # Dino's output
        Y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])


elif data_set == 4:
        X = np.array([
                [10000, 9500,  9000, 9300,
                 9700,  9300,  9700, 9100,
                 8500,  8800,  8600, 8700,
                 9000, 8300, 9100, 8900],

                [12000, 11000, 10550, 10300,
                 12500, 12300, 11500, 11500,
                 10000, 8800, 8600, 9500,
                 9000, 10000, 8500, 8300]])


        X = X / 10000

        xmin = min(X[0])
        xmax = max(X[0])
        ymin = min(X[1])
        ymax = max(X[1])
        X = X.T

        # Dino's output
        Y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])


else:

    X = np.array([
        [10, 9,
         8, 7],

        [10, 9,
         8, 7]])

     # Dino's output
    Y = np.array([0, 0, 1, 1])


    xmin = min(X[0])
    xmax = max(X[0])
    ymin = min(X[1])
    ymax = max(X[1])
    X = X.T



# Fit the data to a logistic regression model.
clf = sklearn.linear_model.LogisticRegression()
clf.fit(X, Y)

# Retrieve the model parameters.
b = clf.intercept_[0]
w1, w2 = clf.coef_.T
# Calculate the intercept and gradient of the decision boundary.
c = -b/w2
m = -w1/w2

# m = -2

print(f"b= {b} ,w1= {w1} ,w2= {w2}")
print(f"m= {m} , c= {c}")

# c= ymax - m * xmin

# print(f"Adjusted c= {c}")

for i, x in enumerate(X):
    predictions = clf.predict(x.reshape(1, -1))
    s = "   <<<---" if predictions[0] != Y[i] else ""
    print(f"{i + 1} x_test={x.T} ,prediction={predictions[0]}, y={Y[i]} {s}")

# Plot the data and the classification with the decision boundary.

xd = np.array([xmin, xmax])
yd = m*xd + c

print(f"xd= {xd}")
print(f"yd= {yd}")

plt.plot(xd, yd, 'k', lw=1, ls='--')
plt.fill_between(xd, yd, ymin, color='tab:blue', alpha=0.2)
plt.fill_between(xd, yd, ymax, color='tab:orange', alpha=0.2)

plt.scatter(*X[Y==0].T, s=40, alpha=0.5)
plt.scatter(*X[Y==1].T, s=40, alpha=0.5)
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.ylabel(r'$x_2$')
plt.xlabel(r'$x_1$')

plt.show()

# plt.savefig('dinos.png')