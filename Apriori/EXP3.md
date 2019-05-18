## 实验三 关联规则挖掘  美国国会投票记录

### 一、实验内容

本题需使用Apriori算法，支持度设为30%，置信度为90%，挖掘高置信度的规则。使用的数据是[美国国会投票纪录](http://archive.ics.uci.edu/ml/datasets/Congressional+Voting+Records)。

### 二、分析及设计

#### 1. 背景

从大规模数据集中寻找物品间的隐含关系被称作关联分析(association analysis)或者关联规则学习(association rule learning)。 关联分析是一种在大规模数据集中寻找有趣关系的任务。这些关系可以有两种形式:频繁项集或者关联规则。频繁项集(frequent item sets)是经常出现在一块的物品的集合,关联规则(association rules)暗示两种物品之间可能存在很强的关系。而对于平凡的定义最重要的是**支持度和可信度**。

**支持度(support): **被定义为数据集中包含该项集的记录所占的比例。支持度是针对项集来说的,因此可以定义一个最小支持度,而只保留满足最小支持度的项集。

**置信度(confidence): **是针对一条诸如{尿布} ➞ {葡萄酒}的关联规则来定义的。这条规则的可信度被定义为“支持度({尿布, 葡萄酒})/支持度({尿布})”。从图11-1中可以看到,由于{尿布, 葡萄酒}的支持度为3/5,尿布的支持度为4/5,所以“尿布 ➞ 葡萄酒”的可信度为3/4=0.75。 这意味着对于包含“尿布”的所有记录,我们的规则对其中75%的记录都适用。为了减少计算量，我们引进Apriori原理来减少计算量。

#### 2. Apriori原理与实现

**原理: **Apriori原理就是说如果某个项集是频繁的，那么它的所有子集都是频繁的，反之，如果某个项集是非频繁的，那么它的所有超集都是非频繁的。这样，当我们计算出某个项集是非频繁的，就不用再计算其超集。这样可以避免项集数目的指数增长，从而在合理时间内算出频繁项集。

**算法代码伪实现: **

> 对数据集中的每条交易记录tran
>
> 对每个候选项集can：
> 	检查一下can是否是tran的子集：
> 	如果是，则增加can的计数值
>
> 对每个候选项集：
> 如果其支持度不低于最小值，则保留该项集
>
> 返回所有频繁项集列表

### 三、详细实现

#### 第一部分

```python
def loadDataSet(self, path):
    return [line.strip().split(',') for line in open(path).readlines()]

# C1 是大小为1的所有候选项集的集合
def createC1(self, dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])  # store all the item unrepeatly

    C1.sort()
    # return list(map(frozenset, C1)) #frozen set, user can't change it.
    return list(map(frozenset, C1))
```

函数 ```loadDataSet()``` 用于导入数据集； 
函数 ```createC1()``` 将构建集合 C1 。 
Apriori算法首先构建集合 C1 ,然后扫描数据集判断这些只有一个元素的项集是否满足最小支持度的要求。那些满足最低要求的项集构成集合 L1 。而 L1 中的元素相互组合构成 C2 , C2 再进一步过滤变为 L2 。

因此算法需要一个函数 ```createC1()``` 来构建第一个候选项集的列表 C1 。由于算法一开始是从输入数据中提取候选项集列表,所以这里需要一个特殊的函数来处理,而后续的项集列表则是按一定的格式存放的。这里使用的格式就是Python中frozenset类型。frozenset是指被“冰冻”的集合,即用户不能修改它们。

#### 第二部分

```python
# 该函数用于从 C1 生成 L1 。
def scanD(self, D, Ck, minSupport):
    # 参数：数据集、候选项集列表 Ck以及感兴趣项集的最小支持度 minSupport
    ssCnt = {}
    for tid in D:  # 遍历数据集
        for can in Ck:  # 遍历候选项
            if can.issubset(tid):  # 判断候选项中是否含数据集的各项
                if not can in ssCnt:
                    ssCnt[can] = 1  # 不含设为1
                else:
                    ssCnt[can] += 1  # 有则计数加1
    numItems = float(len(D))  # 数据集大小
    retList = []  # L1初始化
    supportData = {}  # 记录候选项中各个数据的支持度
    for key in ssCnt:
        support = ssCnt[key] / numItems  # 计算支持度
        if support >= minSupport:
            retList.insert(0, key)  # 满足条件加入L1中
        supportData[key] = support
    return retList, supportData
```

#### 第三部分

```python
# total apriori
def aprioriGen(self, LK, k):  # 组合，向上合并
    # creates Ck 参数：频繁项集列表 Lk 与项集元素个数 k
    retList = []
    lenLk = len(LK)
    for i in range(lenLk):
        for j in range(i + 1, lenLk):  # 两两组合遍历
            L1 = list(LK[i])[:k - 2]
            L2 = list(LK[j])[:k - 2]
            L1.sort()
            L2.sort()
            if L1 == L2:  # 若两个集合的前k-2个项相同时,则将两个集合合并
                retList.append(LK[i] | LK[j])  # set union
    return retList

# apriori
def apriori(self, dataSet, minSupport=0.3):
    C1 = self.createC1(dataSet)
    D = list(map(set, dataSet))  # python3
    L1, supportData = self.scanD(D, C1, minSupport)  # 单项最小支持度判断 0.5，生成L1
    L = [L1]
    k = 2
    while (len(L[k - 2]) > 0):  # 创建包含更大项集的更大列表,直到下一个大的项集为空
        Ck = self.aprioriGen(L[k - 2], k)  # Ck
        Lk, supK = self.scanD(D, Ck, minSupport)  # get Lk
        supportData.update(supK)
        L.append(Lk)
        k += 1
    return L, supportData
```

> 当集合中项的个数大于0时:
>     构建一个k个项组成的候选项集的列表
>     检查数据以确认每个项集都是频繁的
>     保留频繁项集并构建k+1项组成的候选项集的列表(向上合并)

主函数是 ```apriori()``` ,它会调用 ```aprioriGen()``` 来创建候选项集 Ck 。 
函数 ```aprioriGen()``` 的输入参数为频繁项集列表 Lk 与项集元素个数 k ,输出为 Ck 。举例来说,该函数以{0}、{1}、{2}作为输入,会生成{0,1}、{0,2}以及{1,2}。要完成这一点,首先创建一个空列表,然后计算 Lk 中的元素数目。通过循环来比较 Lk 中的每一个元素与其他元素，紧接着,取列表中的两个集合进行比较。如果这两个集合的前面 k-2 个元素都相等,那么就将这两个集合合成一个大小为 k 的集合 。这里使用集合的并操作来完成。

```apriori()```函数首先创建 C1 然后读入数据集将其转化为 D (集合列表)来完 

成。程序中使用 map 函数将 ```set()``` 映射到 dataSet 列表中的每一项。```scanD()``` 函数来创建 L1 ,并将 L1 放入列表 L 中。 L 会包含 L1 、 L2 、 L3 …。现在有了 L1 ,后面会继续找 L2 , L3 …,这可以通过 while 循环来完成,它创建包含更大项集的更大列表,直到下一个大的项集为空。Lk 列表被添加到 L ,同时增加 k 的值,增大项集个数，重复上述过程。最后,当 Lk 为空时,程序返回 L 并退出。

#### 第四部分：从频繁项集中挖掘关联规则

对于关联规则，也有类似的量化方法,这种量化指标称为可信度。一条规则P ➞ H的可信度定义为 support(P |H)/support(P) 。在Python中，操作符 | 表示集合的并操作。P | H 是指所有出现在集合 P 或者集合 H 中的元素。

如果某条规则并不满足最小可信度要求，那么该规则的**所有子集**也不会满足最小可信度要求。那么就可以首先从一个频繁项集开始，接着创建一个规则列表，其中规则右部只包含一个 元素，然后对这些规则进行测试。接下来合并所有剩余规则来创建一个新的规则列表，其中规则右部包含两个元素。

```python
# 生成关联规则
def generateRules(self, L, supportData, minConf=0.9):
    # 频繁项集列表、包含那些频繁项集支持数据的字典、最小可信度阈值
    bigRuleList = []  # 存储所有的关联规则
    for i in range(1, len(L)):  # 只获取有两个或者更多集合的项目，从1,即第二个元素开始，L[0]是单个元素的
        # 两个及以上的才可能有关联一说，单个元素的项集不存在关联问题
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]
            # 该函数遍历L中的每一个频繁项集并对每个频繁项集创建只包含单个元素集合的列表H1
            if (i > 1):
                # 如果频繁项集元素数目超过2,那么会考虑对它做进一步的合并
                self.rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else:  # 第一层时，后件数为1
                self.calcConf(freqSet, H1, supportData, bigRuleList, minConf)  # 调用函数2
    return bigRuleList

# 生成候选规则集合：计算规则的可信度以及找到满足最小可信度要求的规则
def calcConf(self, freqSet, H, supportData, brl, minConf=0.9):
    # 针对项集中只有两个元素时，计算可信度
    prunedH = []  # 返回一个满足最小可信度要求的规则列表
    for conseq in H:  # 后件，遍历 H中的所有项集并计算它们的可信度值
        conf = supportData[freqSet] / supportData[freqSet - conseq]  # 可信度计算，结合支持度数据
        if conf >= minConf:
            # print(freqSet - conseq, '-->', conseq, 'conf:', conf)
            self.sum += 1
            # 如果某条规则满足最小可信度值,那么将这些规则输出到屏幕显示
            brl.append((freqSet - conseq, conseq, conf))  # 添加到规则里，brl 是前面通过检查的 bigRuleList
            prunedH.append(conseq)  # 同样需要放入列表到后面检查
    return prunedH

# 合并
def rulesFromConseq(self, freqSet, H, supportData, brl, minConf=0.9):
    # 参数:一个是频繁项集,另一个是可以出现在规则右部的元素列表 H
    m = len(H[0])
    if (len(freqSet) > (m + 1)):  # 频繁项集元素数目大于单个集合的元素数
        Hmp1 = self.aprioriGen(H, m + 1)  # 存在不同顺序、元素相同的集合，合并具有相同部分的集合
        Hmp1 = self.calcConf(freqSet, Hmp1, supportData, brl, minConf)  # 计算可信度
        if (len(Hmp1) > 1):
            # 满足最小可信度要求的规则列表多于1,则递归来判断是否可以进一步组合这些规则
            self.rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf)
```

#### 第五部分：对于选票数据的处理

通过阅读*house-votes-84.names*文件的信息可以知道，数据包括435条选票记录，共有17个属性，第一列属性为一个布尔值属性，代表你是民主党(democrat)还是共和党(republican)，剩下16列属性都是一些和选票结果息息相关的属性。

![1558117581390](C:\Users\mty19\AppData\Roaming\Typora\typora-user-images\1558117581390.png)

y代表赞成票，能代表反对票，？代表不表态(不是缺失值)。

```python
CQA = ["Class-Name", "handicapped-infants", "water-project-cost-sharing", "adoption-of-the-budget-resolution",
           "physician-fee-freeze", "el-salvador-aid", "religious-groups-in-schools", "anti-satellite-test-ban",
           "aid-to-nicaraguan-contras", "mx-missile", "immigration", "synfuels-corporation-cutback",
           "education-spending",
           "superfund-right-to-sue", "crime", "duty-free-exports", "export-administration-act-south-africa"]

    apriori = Apriori(0)
    path = 'house-votes-84.data'

    voteDataSet = apriori.loadDataSet(path)

    for item in voteDataSet:
        for i in range(17):
            item[i] = CQA[i] + "_" + item[i]
```

我们给每一个属性后面再加上y/n/?的属性，否则挖掘到的规则无法对应到属性上去。

### 四、实验结果

![1558117848882](C:\Users\mty19\AppData\Roaming\Typora\typora-user-images\1558117848882.png)

![1558117883980](C:\Users\mty19\AppData\Roaming\Typora\typora-user-images\1558117883980.png)

图一为和民主党相关的规则，图二又展示出了置信度。

### 五、心得体会

缺点：

- 可能产生庞大的候选集。
- 算法需多次遍历数据集，算法效率低，耗时