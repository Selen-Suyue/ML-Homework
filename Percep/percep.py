import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

"""More imformation about this homerwork at https://github.com/Selen-Suyue/ML-Homework"""

def sign(x):
    """Sign function."""
    return 1 if x >= 0 else -1


def perceptron(X, y, learning_rate=1):
    """Perceptron algorithm with all correct classification as termination condition."""

    n_features = X.shape[1]
    w = np.zeros(n_features)

    epoch = 0
    while True:
        epoch += 1
        misclassified = False
        for i in tqdm(range(X.shape[0]),
                      desc=f"Training Progress - Epoch {epoch}"):

            z = np.dot(w, X[i])

            y_pred = sign(z)

            if y_pred != y[i]:
                w = w + learning_rate * y[i] * X[i]
                misclassified = True

                visualize_and_log(X, y, w, epoch, i)

        if not misclassified:
            break

    return w


def visualize_and_log(X, y, w, epoch, iteration):
    """Visualize decision boundary and log results."""

    plt.figure(figsize=(6, 4))
    plt.scatter(X[y == 1, 1], X[y == 1, 2], marker='o', label='Class 1')
    plt.scatter(X[y == -1, 1], X[y == -1, 2], marker='x', label='Class -1')

    x_values = np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100)
    y_values = -(w[0] + w[1] * x_values) / w[2]
    plt.plot(x_values, y_values, label='Decision Boundary')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f'Epoch {epoch}, Iteration {iteration + 1}')
    plt.legend()
    plt.grid(True)

    plt.savefig(f'epoch_{epoch}_iteration_{iteration + 1}.png')
    plt.close()

    with open('results.txt', 'a') as f:
        f.write(f'Epoch {epoch}, Iteration {iteration + 1}:\n')
        f.write(f'Weights: {w}\n')
        f.write('---\n')


if __name__ == "__main__":
    X = np.array([[1, 1, 2], [1, 2, 1], [1, 2, 3], [1, 1, -1]])
    y = np.array([1, -1, 1, -1])

    w = perceptron(X, y)

    print(f'Final Weights: {w}')

    plt.close('all')
