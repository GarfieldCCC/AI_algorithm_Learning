## 使用学习机(LM)解决字符识别问题

### 一、实验描述

对于一个手写数字数据集：

1. 用无噪声数据训练LM1
2. 将训练好的LM1设置为另一个LM2的初始化，并训练LM2，其中数据集被加上高斯噪声污染
3. 分别测试LM1和LM2，噪声数据在某些范围内变化
4. 绘制识别误差百分比曲线相对于LM1和LM2的噪声水平

### 二、分析及设计

1. **读入数据**

   数据集是一个(47535, 16 x 8)的数据，意味着一共有47535条数据，每一条数据有16 x 8个特征。进一步分析，每一条数据的第一列属性是字母，后面全部是数字0、1，不难想到这其实是一个形状为(16, 8)的矩阵被展成一长串，在原来矩阵中，1代表这个坐标有像素值，否则就没有像素值。这样，就可以通过坐标以及是否有像素值表示一个字母。

   我们将读入的数据分为```data_train```, ```target_train```, ```data_test```, ```target_test```, 以便于后续训练时处理。

2. **对于一个输入Sigma，计算误差**

   这里，我们依次递增输入十个Sigma值，对于每一个Sigma值，我们先获得数据集，然后对数据集加入高斯噪声，接着训练出两个模型，其中一个模型是使用带有高斯噪声的数据进行训练的(即现在高斯噪声数据集上训练一次，再在纯净数据集上训练一次)，另一个模型是使用纯净的数据集进行训练，最后使用带噪声的数据集来评估模型的好坏。

3. **将所有结果汇总并可视化**

4. **关于多层感知机(MLP)**

   多层感知机（MLP，Multilayer Perceptron）也叫人工神经网络（ANN，Artificial Neural Network），除了输入输出层，它中间可以有多个隐层，最简单的MLP只含一个隐层，即三层的结构，如下图![img](https://img-blog.csdn.net/20150128033221168?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMjE2MjYxMw==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/Center)

   从上图可以看到，多层感知机层与层之间是全连接的（全连接的意思就是：上一层的任何一个神经元与下一层的所有神经元都有连接）。多层感知机最底层是输入层，中间是隐藏层，最后是输出层。

### 三、详细实现

1. **读入数据**

```python
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
```

2. **加入高斯噪声**

```python
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
```

3. **训练模型**

* 没有噪声

```python
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

```

* 有噪声

```python
def train_model_with_noise(data_train, target_train, noise_data_train):
    '''
    先对噪声训练集和训练集标签并进行训练
    然后使用无噪声训练集再次训练
    返回训练好的模型
    '''
    model = train_model(noise_data_train, target_train)  # 调用训练函数进行训练
    model.fit(data_train, target_train, epochs=2)  # 使用无噪声训练集再次拟合
    return model  # 返回模型

```

4. **评估模型**

```python
def evaluate_model(model, data_test, target_test):
    '''
    接受训练好的模型，测试集
    返回模型的泛化程度
    '''
    test_loss, test_acc = model.evaluate(data_test, target_test)  # 计算模型在测试集上的准确率
    return test_acc

```

### 四、实验结果

图片中横轴为Sigma的方差，纵轴为accuracy

不适用交叉验证的结果：![Figure_1](E:\File\学校和班级\班级\大三下\机器学习\Task\Figure_1.png)

使用10路交叉验证的结果：![Figure_2](E:\File\学校和班级\班级\大三下\机器学习\Task\Figure_2.png)

### 五、心得体会

1. 随着噪声方差的增加，accuracy的值逐渐减小
2. 经过噪声数据训练的模型对于加入噪声的数据拟合效果更好