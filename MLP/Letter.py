from __future__ import absolute_import, division, print_function
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import numpy as np
import csv, time


def load_data(file_path, n):
    ''''
    file_path接受一个csv文件路径，n接受被分配为测试组的组号
    返回四个数组：训练集，训练集标签，测试集，测试集标签
    '''

    with open(file_path, 'rt') as file:  # 打开以'\t'分隔的csv文件，并读取所有的行
        reader = csv.reader(file, delimiter='\t')
        lines = list(reader)

    data_train, target_train = [], []  # 定义训练集，训练集标签，测试集，测试集标签
    data_test, target_test = [], []

    for line in lines:
        pixels = np.array([int(x) for x in line[6:134]])  # 图像文件为6：134
        pixels = pixels.reshape(16, 8)  # 图像文件的格式为16*8
        if n != int(line[5]):  # 如果组号不等于测试集组号
            data_train.append(pixels)  # 加入训练集
            target_train.append(ord(line[1]) - 97)
        else:
            data_test.append(pixels)  # 否则加入测试集
            target_test.append(ord(line[1]) - 97)
    data_train = np.array(data_train)  # 转化为np.array数组
    data_test = np.array(data_test)
    return data_train, target_train, data_test, target_test


def train_model(data_train, target_train):
    '''
    接受训练集和训练集标签并进行训练
    返回训练好的模型
    '''
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(16, 8)),  # 输入层，接受16*8的数组输入
        keras.layers.Dense(128, activation=tf.nn.relu),  # 隐层，128神经元，激活函数为relu
        keras.layers.Dense(26, activation=tf.nn.softmax)  # 输出层，输出分别输入26个字母的概率
    ])
    model.compile(optimizer=tf.train.AdamOptimizer(),  # 优化器为AdamOptimizer
                  loss='sparse_categorical_crossentropy',  # 损失函数为sparse_categorical_crossentropy
                  metrics=['accuracy'])  # 衡量标准为正确率
    model.fit(data_train, target_train, epochs=2)  # 开始拟合，训练次数为5
    return model  # 返回模型


def train_model_with_noise(data_train, target_train, noise_data_train):
    '''
    先对噪声训练集和训练集标签并进行训练
    然后使用无噪声训练集再次训练
    返回训练好的模型
    '''
    model = train_model(noise_data_train, target_train)  # 调用训练函数进行训练
    model.fit(data_train, target_train, epochs=2)  # 使用无噪声训练集再次拟合
    return model  # 返回模型


def add_gauss_noise(datas, sigma):
    '''
    接受一个数据集并加入方差为sigma的高斯噪声
    返回加入噪声的数据集
    '''
    noise_datas = []
    for data in datas:
        data = data + np.random.rand(16, 8) * sigma  # 加入方差为sigma的高斯噪声并进行调整
        noise_datas.append(data)
    return np.array(noise_datas)  # 返回加入噪声的数据集


def evaluate_model(model, data_test, target_test):
    '''
    接受训练好的模型，测试集
    返回模型的泛化程度
    '''
    test_loss, test_acc = model.evaluate(data_test, target_test)  # 计算模型在测试集上的准确率
    return test_acc


if __name__ == "__main__":
    # 程序启动
    print("Start Training...\n")
    # 定义噪声损失和无噪声损失列表
    clean_acc = []
    noise_acc = []
    # 噪声方差从0-1
    for sigma in range(10):
        c_acc = 0
        n_acc = 0
        # 测试集标号从0-9，即为n路交叉验证
        for i in range(10):
            # 获得数据集
            data_train, target_train, data_test, target_test = load_data('DataSet/letter.csv', i)
            # 加入高斯噪声
            noise_data_train = add_gauss_noise(data_train, sigma / 50)
            noise_data_test = add_gauss_noise(data_test, sigma / 50)
            # 训练
            model_1 = train_model(data_train, target_train)
            model_2 = train_model_with_noise(data_train, target_train, noise_data_train)
            # 计算损失
            n_acc += evaluate_model(model_2, noise_data_test, target_test)
            c_acc += evaluate_model(model_1, noise_data_test, target_test)
        # 记录损失
        noise_acc.append(n_acc / 10)
        clean_acc.append(c_acc / 10)
        print(noise_acc)
        print(clean_acc)
    # 绘制损失折线图
    print("Ending Training...")
    xlabel = [x for x in range(10)]
    plt.plot(xlabel, clean_acc, noise_acc)
    plt.legend(['clean_acc', 'noise_acc'], loc='upper right', r=['r', 'b'])
    plt.show()
