import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv('Grand-slams-men-2013.csv', encoding='latin1')

x = dataset['NPA.1']
y = dataset['UFE.2']


def cubic_reg(x, y, knots, color):
    x_a = x.values
    y_a = y.values

    idx = x_a.argsort()
    x_a = x_a[idx]
    y_a = y_a[idx]

    o = np.ones((239, 1))

    xi = np.array([x_a]).transpose()
    xi_2 = np.power(xi, 2)
    xi_3 = np.power(xi, 3)

    knot_columns = []
    for knot in knots:
        results = []
        for x in np.nditer(xi):
            result = x - knot
            if (result < 0):
                result = 0
            results.append(result)
        col = np.array([results]).transpose()
        col_3 = np.power(col, 3)
        knot_columns.append(col_3)

    X = 0

    if (len(knot_columns) == 1):
        X = np.hstack((o, xi, xi_2, xi_3, knot_columns[0]))
    elif (len(knot_columns) == 2):
        X = np.hstack((o, xi, xi_2, xi_3, knot_columns[0], knot_columns[1]))
    elif (len(knot_columns) == 3):
        X = np.hstack((o, xi, xi_2, xi_3, knot_columns[0], knot_columns[1], knot_columns[2]))

    X_t = X.transpose()
    Ba = np.dot(X_t, X)
    Bb = np.linalg.inv(Ba).dot(X_t)
    B = np.dot(Bb, y_a)
    reg = X.dot(B)
    print(X.flatten())
    plt.scatter(x_a, y_a, color="pink")

    plt.plot(x_a, reg, color=color)



def main():
    cubic_reg(x, y, [30], "blue")
    cubic_reg(x, y, [15, 30], "red")
    cubic_reg(x, y, [10, 20, 30], "green")
    plt.show()


if __name__ == "__main__":
    main()
