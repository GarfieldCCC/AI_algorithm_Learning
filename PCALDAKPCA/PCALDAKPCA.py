from sklearn.datasets import load_iris
import numpy as np
from sklearn import decomposition
import matplotlib.pyplot as plt


def load_data():
    return load_iris()


def load_data_tag():
    iris = load_data()
    return iris.data, iris.target


def PCA(data, n):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n)
    pca_result = pca.fit_transform(data.data)
    return pca_result


def LDA(data, n):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    lda = LDA(n_components=n)
    lda_result = lda.fit_transform(data.data, data.target)
    return lda_result


def plot(data, n):
    pca_result = PCA(data, n)
    lda_result = LDA(data, n)

    plt.subplot(1, 2, 1)
    plt.scatter(pca_result[data.target == 0, 0], pca_result[data.target == 0, 1], color='r')
    plt.scatter(pca_result[data.target == 1, 0], pca_result[data.target == 1, 1], color='g')
    plt.scatter(pca_result[data.target == 2, 0], pca_result[data.target == 2, 1], color='b')
    plt.title('PCA on iris')

    plt.subplot(1, 2, 2)
    plt.scatter(lda_result[data.target == 0, 0], lda_result[data.target == 0, 1], color='r')
    plt.scatter(lda_result[data.target == 1, 0], lda_result[data.target == 1, 1], color='g')
    plt.scatter(lda_result[data.target == 2, 0], lda_result[data.target == 2, 1], color='b')
    plt.title('LDA on iris')

    plt.show()


def plot_KPCA(*data):
    X, y = data
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    fig = plt.figure()

    for i, kernel in enumerate(kernels):
        kpca = decomposition.KernelPCA(n_components=2, kernel=kernel)
        kpca.fit(X)
        X_r = kpca.transform(X)
        ax = fig.add_subplot(2, 2, i + 1)
        for label in np.unique(y):
            position = y == label
            ax.scatter(X_r[position, 0], X_r[position, 1], label="target=%d" % label)
            ax.set_xlabel('x[0]')
            ax.set_ylabel('x[1]')
            ax.legend(loc='best')
            ax.set_title('kernel=%s' % kernel)
    plt.suptitle("KPCA")
    plt.show()


if __name__ == '__main__':
    iris = load_data()
    X, y = load_data_tag()
    n_components = 2

    plot(iris, n_components)
    plot_KPCA(X, y)
