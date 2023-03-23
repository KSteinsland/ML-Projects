import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn


def normalize(df, rl, rh):
    """Normalize columns in dataframe to range [rl, rh]"""
    dtypes = df.dtypes
    for i in range(len(df.columns)):
        if dtypes[i] in ['float64']:
            ma = max(df[i])
            mi = min(df[i])
            df[i] = (np.array(df[i].tolist()) - mi) * (rh - rl) / (ma - mi) + rl
    return df


def divide(data, first_batch_size, sec_batch_size, numb_variants):
    """Divide data into two batches"""
    size = first_batch_size + sec_batch_size
    assert size == len(data[:, 0])/numb_variants

    variants = [data[i * size: (i+1)*size, :] for i in range(numb_variants)]

    first_batch = [v[:first_batch_size] for v in variants]
    sec_batch = [v[first_batch_size:] for v in variants]

    return np.array(first_batch).reshape((numb_variants*first_batch_size, -1)), \
           np.array(sec_batch).reshape((numb_variants*sec_batch_size, -1))


def sigma(x):
    """Sigmoid function"""
    return 1 / (1 + np.exp(-x))


def train(data, numb_iterations, alpha, numb_classes):
    """Trains a linear classifier with gradient descent"""
    numb_features = len(data[0]) - 1
    MSEs = []
    W = np.zeros((numb_classes, numb_features + 1))

    for m in range(numb_iterations):
        MSE = 0
        nabla_MSE = 0

        for d in data:
            x_k = np.ones(numb_features + 1)
            x_k[:numb_features] = d[:numb_features]

            t_k = np.zeros(numb_classes)
            t_k[int(d[-1])] = 1

            g_k = sigma(W @ x_k)

            MSE += (g_k - t_k).reshape((1, numb_classes)) @ \
                   (g_k - t_k).reshape((numb_classes, 1))
            nabla_MSE += ((g_k - t_k) * g_k * (1 - g_k)).reshape((numb_classes, 1)) @ \
                        x_k.reshape((1, numb_features + 1))

        W = W - alpha * nabla_MSE
        MSEs.append(float(MSE))

    return W, MSEs


def classify(W, data):
    """Classifies data based on weighting matrix W"""
    numb_features = len(data[0]) - 1
    variants = []

    for d in data:
        x_k = np.ones((numb_features + 1))
        x_k[:numb_features] = d[:numb_features]
        x_k.reshape((numb_features + 1, 1))

        g = sigma(W @ x_k)
        variant = list(g).index(max(g))
        variants.append(variant)

    return variants


def conf_matrix(y_true, y_pred, numb_classes):
    """Calculates the weighting matrix C"""
    C = np.zeros((numb_classes, numb_classes))

    for i in range(len(y_true)):
        C[y_true[i], y_pred[i]] += 1

    row_sum = C.sum(axis=1)
    return C / row_sum[:, None]


def error_rate(y_true, y_pred):
    """Calculates error rate"""
    false_predictions = 0
    for i in range(len(y_true)):
        if y_true[i] != y_pred[i]:
            false_predictions += 1
    return false_predictions / len(y_true)


if __name__ == '__main__':
    # constants
    classes = 3
    features = 4
    labels = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
    iterations = 5000
    alpha = 0.2

    # import data and normalize
    iris_df = pd.read_csv("iris.data", header=None)
    norm_range_low = 0
    norm_range_high = 1
    iris_df_nor = normalize(iris_df, norm_range_low, norm_range_high)

    data = np.array(iris_df_nor.values)

    # convert variant to number
    d = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
    data[:, 4] = [d[x] for x in data[:, 4]]


    # part 1 -------------------------------------------------------------------

    # divide data.
    # first 30 as training data
    training_data, testing_data = divide(data.copy(), 30, 20, classes)
    # last 30 as training data, (uncomment line below)
    #testing_data, training_data = divide(data.copy(), 20, 30, classes)

    # train
    W, mse_list = train(training_data, iterations, alpha, classes)

    # test on training data
    predicted_variants = classify(W, training_data)
    true_variants = [int(x) for x in training_data[:, -1]]

    # confusion matrix
    C = conf_matrix(true_variants, predicted_variants, classes)
    fig, axes = plt.subplots(2, 1)
    axes[0].set_title("Training data")
    ax = sn.heatmap(C, annot=True, fmt=".2f", cmap="viridis", xticklabels=labels,
                    yticklabels=labels, ax=axes[0])
    ax.set(xlabel='Predicted', ylabel='True')

    # error rate
    e = error_rate(true_variants, predicted_variants)
    print("error rate:", round(e, 2) * 100, "%")

    # test on testing data
    predicted_variants = classify(W, testing_data)
    true_variants = [int(x) for x in testing_data[:, -1]]

    # confusion matrix
    C = conf_matrix(true_variants, predicted_variants, classes)
    axes[1].set_title("Testing data")
    ax = sn.heatmap(C, annot=True, fmt=".2f", cmap="viridis", xticklabels=labels,
                    yticklabels=labels, ax=axes[1])
    ax.set(xlabel='Predicted', ylabel='True')
    plt.show()

    # error rate
    e = error_rate(true_variants, predicted_variants)
    print("error rate:", e * 100, "%")


    # part 2 -------------------------------------------------------------------

    # plot histogram
    iris_df.columns = ['Sepal length [cm]', 'Sepal width [cm]', 'Petal Length [cm]',
                       'Petal Width [cm]', "Species"]
    fig, axes = plt.subplots(2, 2, figsize=(8, 9))

    for i in range(len(iris_df.columns) - 1):
        sn.histplot(ax=axes[i // 2, i % 2], data=iris_df, x=iris_df.columns[i],
                    hue=iris_df.columns[-1], bins=15)
    plt.show()

    # first 30 as training data
    training_data, testing_data = divide(data.copy(), 30, 20, classes)

    fig, axes = plt.subplots(4, 1, figsize=(7, 12))

    # no feature removed
    W, mse_list = train(training_data, iterations, alpha, classes)
    predicted_variants = classify(W, testing_data)
    true_variants = [int(x) for x in testing_data[:, -1]]
    C = conf_matrix(true_variants, predicted_variants, classes)
    axes[0].set_title("no feature removed")
    ax = sn.heatmap(C, annot=True, fmt=".2f", cmap="viridis", xticklabels=labels,
                    yticklabels=labels, ax=axes[0])
    ax.set(xlabel='Predicted', ylabel='True')
    e = error_rate(true_variants, predicted_variants)
    print("error rate:", e * 100, "%")

    # removing sepal width
    training_data, testing_data = np.delete(training_data, 1, 1), np.delete(testing_data, 1, 1)
    W, mse_list = train(training_data, iterations, alpha, classes)
    predicted_variants = classify(W, testing_data)
    true_variants = [int(x) for x in testing_data[:, -1]]
    C = conf_matrix(true_variants, predicted_variants, classes)
    axes[1].set_title("removed sepal width")
    ax = sn.heatmap(C, annot=True, fmt=".2f", cmap="viridis", xticklabels=labels,
                    yticklabels=labels, ax=axes[1])
    ax.set(xlabel='Predicted', ylabel='True')
    e = error_rate(true_variants, predicted_variants)
    print("error rate:", e * 100, "%")

    # removing sepal length
    training_data, testing_data = np.delete(training_data, 0, 1), np.delete(testing_data, 0, 1)
    W, mse_list = train(training_data, iterations, alpha, classes)
    predicted_variants = classify(W, testing_data)
    true_variants = [int(x) for x in testing_data[:, -1]]
    C = conf_matrix(true_variants, predicted_variants, classes)
    axes[2].set_title("removed sepal width and sepal length")
    ax = sn.heatmap(C, annot=True, fmt=".2f", cmap="viridis", xticklabels=labels,
                    yticklabels=labels, ax=axes[2])
    ax.set(xlabel='Predicted', ylabel='True')
    e = error_rate(true_variants, predicted_variants)
    print("error rate:", round(e, 2) * 100, "%")

    # removing Petal length
    training_data, testing_data = np.delete(training_data, 0, 1), np.delete(testing_data, 0, 1)
    W, mse_list = train(training_data, iterations, alpha, classes)
    predicted_variants = classify(W, testing_data)
    true_variants = [int(x) for x in testing_data[:, -1]]
    C = conf_matrix(true_variants, predicted_variants, classes)
    axes[3].set_title("removed sepal width, sepal length and petal length")
    ax = sn.heatmap(C, annot=True, fmt=".2f", cmap="viridis", xticklabels=labels,
                    yticklabels=labels, ax=axes[3])
    ax.set(xlabel='Predicted', ylabel='True')
    e = error_rate(true_variants, predicted_variants)
    print("error rate:", round(e, 2) * 100, "%")

    plt.show()
