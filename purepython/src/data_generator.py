import numpy as np
import tensorflow as tf
from sklearn import datasets
from tensorflow.examples.tutorials.mnist import input_data


def convert_to_1_hot(in_data, max_val):
    a = np.array(in_data)
    out_data = np.zeros((len(in_data),max_val))
    out_data[np.arange(len(in_data)), a] = 1
    return out_data


def get_iris(train_test_ratio):
    iris = datasets.load_iris()
    x = iris.data
    y = convert_to_1_hot(iris.target, 3)

    cutoff = int(x.shape[0] * train_test_ratio)

    x_train = x[0:cutoff, :]
    x_test = x[cutoff:, :]
    y_train = y[:cutoff]
    y_test = y[cutoff:]
    return x_train, y_train, x_test, y_test


def get_mnist(train_test_ratio):
    # mnist = tf.examples.tutorials.mnist.input_data.read_data_sets("MNIST_data/", one_hot=True)
    digits = datasets.load_digits()
    x = digits.data

    y = convert_to_1_hot(digits.target, 10)
    cutoff = int(x.shape[0] * train_test_ratio)

    x_train = x[0:cutoff, :] / 16.
    x_test = x[cutoff:, :] / 16.
    y_train = y[0:cutoff]
    y_test = y[cutoff:]
    return x_train, y_train, x_test, y_test


def generate_noisy_polinomial_data(mu, sigma, n_samples, f0, f1, f2, low=-100., high=100. ):
    if mu == 0. and sigma == 0.:
        noise = np.zeros(n_samples)
    else:
        noise = np.random.normal(mu, sigma, n_samples)

    x = (high - low) * np.random.random_sample(n_samples) + low

    y = f0 * x**2 + f1 * x**1 + f2 + noise
    x = np.expand_dims(x, 1)
    y = np.expand_dims(y, 1)

    return x, y


def generate_noisy_linear_data(mu,sigma,n_samples, w, b, low=-100.,high=100.):
    if mu == 0. and sigma == 0.:
        noise = np.zeros(n_samples)
    else:
        noise = np.random.normal(mu, sigma, n_samples)
    x = (high - low) * np.random.random_sample(n_samples) + low

    y = x * w + b + noise

    x = np.expand_dims(x,1)
    y = np.expand_dims(y,1)

    return x, y



if __name__ == '__main__':
    mnist = get_mnist()