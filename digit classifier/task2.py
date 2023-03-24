import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from pathlib import Path
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

def get_data(path):
    ## load data to numpy array
    with open(path, "rb") as f:
        dat = f.read()
    return np.frombuffer(dat, dtype=np.uint8).copy()

def get_data_mnist(dir_path, train_size=60000, test_size=10000):
    ## load mnist train and test data
    p = Path(dir_path)
    test_path = list(p.glob('test_*'))
    train_path = list(p.glob('train_*'))

    d = 784

    x_test = 1.0 / 255 * get_data(test_path[0])[0x10:].reshape((10000,d))
    y_test = get_data(test_path[1])[0x08:].reshape((10000, 1))
    x_test = x_test[:test_size, :]
    y_test = y_test[:test_size, :]

    x_train = 1.0 / 255 * get_data(train_path[0])[0x10:].reshape((60000, d))
    y_train = get_data(train_path[1])[0x08:].reshape((60000, 1))
    x_train = x_train[:train_size,:]
    y_train = y_train[:train_size, :]

    return x_train, y_train, x_test, y_test


def conf_matrix(y_true, y_pred, num_classes=10):
    
    C = np.zeros((num_classes, num_classes))

    for i in range(len(y_true)):
        C[y_true[i],y_pred[i]] += 1

    row_sum = C.sum(axis=1)
    
    return C / row_sum[:, None]


def plot_conf_matrix(C, name="output"):
    ax = sn.heatmap(C, annot=True, fmt=".2f", cmap="viridis")
    ax.set(xlabel='Predicted', ylabel='True')
    ax.set(title=name)
    plt.show()
    fig = ax.get_figure()
    fig.savefig(name + ".eps")


def plot_image(x, y_pred, y_true, name="plot"):
    ax = plt.imshow(x[:].reshape((28,28)))
    plt.title("Prediction " + str(y_pred) + ", True " + str(y_true))
    plt.show()
    fig = ax.get_figure()
    fig.savefig(name + ".eps")


def plot_images(x, y_pred, y_true, name="plot"):
    fig = plt.figure()
    for i in range(x.shape[0]):
        ax = plt.subplot(2,4,i+1,frameon=False, xticks=[], yticks=[])
        plt.imshow(x[i,:].reshape((28,28)))
        plt.title("True " + str(y_true[i]) + ", Pred " + str(y_pred[i]))

    plt.show()
    fig.savefig(name + ".eps")


def dist(A,B):
    return cdist(A, B).T


def error_rate(y_true, y_pred):
    return 1 - np.sum(y_true[:]==y_pred[:])/y_true.shape[0]


def predict_nn(ref, ref_labels, x):
    distances = dist(ref, x)
    nearest = np.argmin(distances, axis=1)
    return ref_labels[nearest[:]]


def mayority_vote(k_nearest_i, k):
    ## finds the class of data point according to a majority vote

    k_nearest_distances_i = k_nearest_i[:k]
    k_nearest_classes_i = k_nearest_i[k:]

    # get the different classes that are present in the k nearest region, and their count
    (unique_classes, unique_classes_counts) = np.unique(k_nearest_classes_i, return_counts=True)

    # get the count of the how many classes has a specific count  
    different_counts = np.bincount(unique_classes_counts)

    if different_counts[-1] == 1: # there is only one class with the highest count
        k_class = unique_classes[np.argmax(unique_classes_counts)]

    else:
        # tie between classes

        # find the classes that are at a tie
        tie_classes = unique_classes[unique_classes_counts==len(different_counts)-1]

        # sort the classes by distance
        k_sorted_indexes = np.argsort(k_nearest_distances_i)
        
        # return the class with the smallest distance 
        # that are also in the tie group
        for i in range(k_sorted_indexes.shape[0]):
            nearest_class = k_nearest_classes_i[k_sorted_indexes[i]] 
            if nearest_class in tie_classes:
                k_class = nearest_class
                break  

    return int(k_class)


def predict_knn(ref, ref_labels, x, k):
    distances = dist(ref, x)

    # find the classes of the k nearest references, not sorted
    k_nearest_indexes = np.argpartition(distances, k, axis=1)[:,:k]
    k_nearest_classes = ref_labels[k_nearest_indexes[:],0]

    # extract the distances from the k nearest refernces
    k_nearest_distances = distances[np.arange(distances.shape[0])[:, None], k_nearest_indexes]

    k_nearest_data = np.concatenate((k_nearest_distances, k_nearest_classes), axis=1)
    
    # predict classes according to majority vote
    predictions = np.apply_along_axis(mayority_vote, 1, k_nearest_data, k)
    return predictions.reshape(-1,1)



# Load the mnist data set
dir_path = "./data/MNIST/"
x_train, y_train, x_test, y_test = get_data_mnist(dir_path)

num_classes = 10
data_size = 784

# Check if data looks correct
plot_image(x_test[0,:], y_test[0,0], y_test[0,0])


# Extract a chunk used for the first part of the task
# as the computation time using the whole set is too large
x_train_chunk = x_train[:30000,:]
y_train_chunk = y_train[:30000,:]

x_test_chunk = x_test[:5000,:]
y_test_chunk = y_test[:5000,:]


## Generate templates by clustering

M = 64 # Number of clusters per class
kmeans = KMeans(
n_clusters=M,
init="k-means++",
n_init=10,
random_state=52342, # for reproducibility
max_iter=100)

template_data = np.zeros((M * num_classes, data_size))
template_labels = np.floor_divide(np.arange(M*num_classes), M).reshape(-1,1)

for i in range(num_classes):
    # extract all points corresponding to class i
    training_class_features = x_train[np.argwhere(y_train.reshape(-1) == i)[:,0]]
    
    # Cluster the class data
    kmeans.fit(training_class_features)
    template_data[M * i: M * (i + 1), :] = kmeans.cluster_centers_


## NN prediction using a chunk of the data set
y_pred_nn = predict_nn(x_train_chunk, y_train_chunk, x_test_chunk)
error_rate_nn = error_rate(y_test_chunk, y_pred_nn)
print("NN error rate", error_rate_nn)

## NN predicition using clustered data set as templates
y_pred_nn_cluster = predict_nn(template_data, template_labels, x_test)
error_rate_nn_cluster = error_rate(y_test, y_pred_nn_cluster)
print("NN w clustering error rate", error_rate_nn_cluster)

## k-NN prediction using clustered data set as templates
k = 7
y_pred_knn = predict_knn(template_data, template_labels, x_test, k)
error_rate_knn_cluster = error_rate(y_test, y_pred_knn)
print("k-NN w clustering error rate", error_rate_knn_cluster)


## Plot confusion matrices for the different classifiers
C = conf_matrix(y_test_chunk, y_pred_nn)
plot_conf_matrix(C, "NN Classifier")

C_cluster = conf_matrix(y_test, y_pred_nn_cluster)
plot_conf_matrix(C_cluster, "NN Classifier, with clustering")

C_knn = conf_matrix(y_test, y_pred_knn)
plot_conf_matrix(C_knn, "KNN Classifier, with clustering")


## Plot some correctly and incorrectly classified pictures using 
# the NN classifier on the training set

# Extract indices where the true label equals number
numbers=[4,8,9]
true_label_eq_num = (y_test_chunk==None).reshape(-1)
for number in numbers:
    true_equals_number = (y_test_chunk==number).reshape(-1)
    true_label_eq_num = np.logical_or(true_label_eq_num, true_equals_number)


# Find all wrong predictions and extract those with the number of interest
w_pred_nn = (y_test_chunk[:]!=y_pred_nn[:]).reshape(-1)
w_pred_nn = np.logical_and(true_label_eq_num, w_pred_nn)

w_labels_nn = y_pred_nn[np.where(w_pred_nn)]
c_labels_nn = y_test_chunk[np.where(w_pred_nn)]
w_pitcures_nn = x_test_chunk[np.where(w_pred_nn),:][0,:,:]

ind = 17
plot_images(w_pitcures_nn[ind:ind+8,:], w_labels_nn[ind:ind+8,0], 
            c_labels_nn[ind:ind+8,0], "incorrect")


# Find all correct predictions and extract those with the number of interest
c_pred_nn = (y_test_chunk[:]==y_pred_nn[:]).reshape(-1)
c_pred_nn = np.logical_and(true_label_eq_num, c_pred_nn)

c_pred_labels_nn = y_pred_nn[np.where(c_pred_nn)]
c_labels_nn = y_test_chunk[np.where(c_pred_nn)]
c_pitcures_nn = x_test_chunk[np.where(c_pred_nn),:][0,:,:]

ind = 17
plot_images(c_pitcures_nn[ind:ind+8,:], c_pred_labels_nn[ind:ind+8,0],
            c_labels_nn[ind:ind+8,0], "correct")