from numpy import *


class Apriori:

    def __init__(self, sum):
        self.sum = sum

    def loadDataSet(self, path):
        # return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]
        # return [[1, 2, 3], [1, 2, 4], [1, 3, 4], [1, 2, 3, 5], [1, 3, 5], [2, 4, 5], [1, 2, 3, 4]]

        return [line.strip().split(',') for line in open(path).readlines()]

    # C1 是大小为1的所有候选项集的集合
    def createC1(self, dataSet):
        C1 = []
        for transaction in dataSet:
            for item in transaction:
                if not [item] in C1:
                    C1.append([item])  # store all the item unrepeatly

        C1.sort()
        # return map(frozenset, C1)#frozen set, user can't change it.
        return list(map(frozenset, C1))

    # 该函数用于从 C1 生成 L1 。
    def scanD(self, D, Ck, minSupport):
        # 参数：数据集、候选项集列表 Ck以及感兴趣项集的最小支持度 minSupport
        ssCnt = {}  # key: 是数据集子集的候选集, value: 这个候选集出现的次数
        for tid in D:  # 遍历数据集
            for can in Ck:  # 遍历候选项
                if can.issubset(tid):  # 判断候选项中是否含数据集的各项
                    # if not ssCnt.has_key(can): # python3 can not support
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

        # 返回满足条件的候选集和对应的支持度
        return retList, supportData

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
    def apriori(self, dataSet, minSupport=0.5):
        C1 = self.createC1(dataSet)
        D = list(map(set, dataSet))  # python3
        L1, supportData = self.scanD(D, C1, minSupport)  # 单项最小支持度判断 0.5，生成L1
        # print(len(L1), L1)
        L = [L1]
        k = 2
        while (len(L[k - 2]) > 0):  # 创建包含更大项集的更大列表,直到下一个大的项集为空
            Ck = self.aprioriGen(L[k - 2], k)  # Ck # 组合，向上合并
            Lk, supK = self.scanD(D, Ck, minSupport)  # get Lk # 返回满足条件的候选集和对应的支持度
            supportData.update(supK)
            L.append(Lk)
            k += 1
        return L, supportData

    # 生成关联规则
    def generateRules(self, L, supportData, minConf=0.9):
        # 频繁项集列表、包含那些频繁项集支持数据的字典、最小可信度阈值
        bigRuleList = []  # 存储所有的关联规则
        for i in range(1, len(L)):  # 只获取有两个或者更多集合的项目，从1,即第二个元素开始，L[0]是单个元素的
            # 两个及以上的才可能有关联一说，单个元素的项集不存在关联问题
            for freqSet in L[i]:  # 对于每一个频繁项集
                H1 = [frozenset([item]) for item in freqSet]
                print("freqSet: ", freqSet)
                print("H1: ", H1)
                # 该函数遍历L中的每一个频繁项集并对每个频繁项集创建只包含单个元素集合的列表H1
                if (i > 1):
                    # 如果频繁项集元素数目超过2,那么会考虑对它做进一步的合并
                    self.rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
                else:  # 第一层时，后件数为1
                    self.calcConf(freqSet, H1, supportData, bigRuleList, minConf)  # 调用函数2
        return bigRuleList

    # 生成候选规则集合：计算规则的可信度以及找到满足最小可信度要求的规则
    def calcConf(self, freqSet, H, supportData, brl, minConf=0.9):
        # print("calcConf-freqSet: ", freqSet)
        # print("calcConf-H: ", H)
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


def main():
    CQA = ["Class-Name", "handicapped-infants", "water-project-cost-sharing", "adoption-of-the-budget-resolution",
           "physician-fee-freeze", "el-salvador-aid", "religious-groups-in-schools", "anti-satellite-test-ban",
           "aid-to-nicaraguan-contras", "mx-missile", "immigration", "synfuels-corporation-cutback",
           "education-spending",
           "superfund-right-to-sue", "crime", "duty-free-exports", "export-administration-act-south-africa"]

    apriori = Apriori(0)
    path = 'house-votes-84.data'

    dataset = [[1, 2, 3],
               [2, 4],
               [2, 3],
               [1, 2, 4],
               [1, 3],
               [2, 3],
               [1, 3],
               [1, 2, 3, 5],
               [1, 2, 3]]


    voteDataSet = apriori.loadDataSet(path)

    for item in voteDataSet:
        for i in range(17):
            item[i] = CQA[i] + "_" + item[i]

    L, suppData = apriori.apriori(dataset, minSupport=0.3)
    rules = apriori.generateRules(L, suppData, minConf=0.5)

    # for item in L:
    #     for i in item:
    #         if 'Class-Name_democrat' in i:
    #             print(i)
    # print(apriori.sum)


if __name__ == '__main__':
    main()

# https://blog.csdn.net/sinat_17196995/article/details/71124284
