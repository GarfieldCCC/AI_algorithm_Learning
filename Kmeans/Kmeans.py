import numpy as np
import matplotlib.pyplot as plt


# Load Data
def loadDataSet(fileName):
    data = np.loadtxt(fileName, delimiter='\t')
    return data


# Euclidean distance calculation
def distEclud(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


# Construct a collection of K random centroids for a given data set
def randCent(dataSet, k):
    m, n = dataSet.shape
    centroids = np.zeros((k, n))
    for i in range(k):
        index = int(np.random.uniform(0, m))  #
        centroids[i, :] = dataSet[index, :]
    return centroids


# K-means clustering
def KMeans(dataSet, k):
    m = np.shape(dataSet)[0]  # Number of rows
    # Which cluster does the first column of the sample belong to?
    # Error in the second column of the sample to the center point of the cluster
    clusterAssment = np.mat(np.zeros((m, 2)))
    clusterError = True

    # 第1步 初始化centroids
    centroids = randCent(dataSet, k)
    while clusterError:
        clusterError = False

        # Traverse all samples (number of rows)
        for i in range(m):
            minDist = 100000.0
            minIndex = -1

            # Traverse all centroids
            # 第2步 找出最近的质心
            for j in range(k):
                # Calculate the Euclidean distance from the sample to the centroid
                distance = distEclud(centroids[j, :], dataSet[i, :])
                if distance < minDist:
                    minDist = distance
                    minIndex = j
            # 第 3 步：更新每一行样本所属的簇
            if clusterAssment[i, 0] != minIndex:
                clusterError = True
                clusterAssment[i, :] = minIndex, minDist ** 2
        # 第 4 步：更新质心
        for j in range(k):
            pointsInCluster = dataSet[np.nonzero(clusterAssment[:, 0].A == j)[0]]  # 获取簇类所有的点
            centroids[j, :] = np.mean(pointsInCluster, axis=0)  # 对矩阵的行求均值

    print("Congratulations,cluster complete!")
    return centroids, clusterAssment


def showCluster(dataSet, k, centroids, clusterAssment):
    m, n = dataSet.shape
    if n != 2:
        print("Data is not two-dimensional")
        return 1

    mark = ['or', 'ob', 'og', 'oy', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    if k > len(mark):
        print("k value is too big")
        return 1

    # 绘制所有的样本
    for i in range(m):
        markIndex = int(clusterAssment[i, 0])
        plt.plot(dataSet[i, 0], dataSet[i, 1], mark[markIndex])

    mark = ['Dr', 'Db', 'Dg', 'Dy', '^b', '+b', 'sb', 'db', '<b', 'pb']
    # 绘制质心
    for i in range(k):
        plt.plot(centroids[i, 0], centroids[i, 1], mark[i])

    plt.show()


dataSet = loadDataSet("testSet.txt")
k = 6
centroids, clusterAssment = KMeans(dataSet, k)

showCluster(dataSet, k, centroids, clusterAssment)
