import numpy as np
import pandas as pd
from sklearn import datasets
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D


def load_iris():
    iris = datasets.load_iris()
    R = np.array(iris.data)
    # print(R)
    return R, iris


def covmat(R, data):
    R_cov = np.cov(R, rowvar=False)
    iris_covmat = pd.DataFrame(data=R_cov, columns=data.feature_names)
    iris_covmat.index = data.feature_names
    # print(iris_covmat)
    return iris_covmat, R_cov


def eig(R_cov):
    eig_values, eig_vectors = np.linalg.eig(R_cov)
    # print(eig_values)
    # print(eig_vectors)
    return eig_values, eig_vectors


def featureExtraction(eig_vectors, R, data, n):
    featureVector = eig_vectors[:, :n]
    # print(featureVector)

    featureVector_t = np.transpose(featureVector)
    R_t = np.transpose(R)
    newDataset_t = np.matmul(featureVector_t, R_t)
    newDataset = np.transpose(newDataset_t)
    # print(newDataset.shape)

    if n == 3:
        df = pd.DataFrame(data=newDataset, columns=['PC1', 'PC2', 'PC3'])
        y = pd.Series(data.target)
        y = y.replace(0, 'setosa')
        y = y.replace(1, 'versicolor')
        y = y.replace(2, 'virginica')
        df['Target'] = y
        # print(df)

        C1 = df[df['Target'] == 'setosa'].index.tolist()[-1]
        C2 = df[df['Target'] == 'versicolor'].index.tolist()[-1]
        C3 = df[df['Target'] == 'virginica'].index.tolist()[-1]

        ax = plt.subplot(111, projection='3d')
        X = df[:C1 + 1]
        Y = df[C1 + 1:C2 + 1]
        Z = df[C2 + 1:C3 + 1]

        ax.scatter(X['PC1'], X['PC2'], X['PC3'], color='r')
        ax.scatter(Y['PC1'], Y['PC2'], Y['PC3'], color='y')
        ax.scatter(Z['PC1'], Z['PC2'], Z['PC3'], color='b')

        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')

        plt.legend(['setosa', 'versicolor', 'virginica'], loc='upper right')
        plt.title('3-Dimension')

        plt.show()
    elif n == 2:
        df = pd.DataFrame(data=newDataset, columns=['PC1', 'PC2'])
        y = pd.Series(data.target)
        y = y.replace(0, 'setosa')
        y = y.replace(1, 'versicolor')
        y = y.replace(2, 'virginica')
        df['Target'] = y
        # print(df)

        sns.lmplot(x='PC1', y='PC2', data=df, hue='Target', fit_reg=False, legend=True)
        plt.title('2-Dimension')
        plt.show()


def main():
    R, iris = load_iris()
    iris_covmat, R_cov = covmat(R, iris)
    eig_values, eig_vectors = eig(R_cov)
    featureExtraction(eig_vectors, R, iris, 2)
    featureExtraction(eig_vectors, R, iris, 3)


if __name__ == '__main__':
    main()

'''
https://www.kaggle.com/lalitharajesh/iris-dataset-exploratory-data-analysis
https://blog.csdn.net/Eddy_zheng/article/details/48713449
https://blog.csdn.net/weixin_39750084/article/details/81952008
https://blog.csdn.net/wanglingli95/article/details/78887771
https://zhuanlan.zhihu.com/p/23636308
'''
