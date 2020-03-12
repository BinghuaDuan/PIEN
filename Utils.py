#===============================================================
#读取CSV文件的方法
import csv
import os

def read_csv(path , dimention):
    """
    :param path: csv文件的路径
    :param dimention: 返回的文件内容列表的维度，
    :return: content_list, path对应的文件不存在，则创建文件,并返回为空，否则，
    若dimension==1，则将csv文件中的各行内容连接起来，
    并返回一维的content_list[...]；
    若dimension==2，则将csv文件中的每一行作为一个单独的列表，
    返回二维的content_list[[...],[...]]；
    """
    content_list = []
    if os.path.exists(path):
        with open(path, mode='r', encoding='utf-8') as fp:
            reader = csv.reader(fp)
            for line in reader:

                if dimention == 1:
                    content_list.extend(line)
                else:
                    content_list.append(line)
    else:
        fp = open(path, mode='w+')
        fp.close()

    return content_list


def write_csv(path, items, mode_str):
    """
    :param path: 写入的文件路径，type:str
    :param items: 写入的文件内容，type:list
    :return: None
    """
    with open(path, mode=mode_str, encoding='utf-8', newline='') as fp:
        writer = csv.writer(fp)
        for item in items:
            if type(item) == list:
                writer.writerow(item)
            else:
                writer.writerow([item])

#==================================================================
"""
建立时间：2019/3/24 10：54
文件描述：将每一个轨道序列抽象为一个类，在类里面实现对轨道序列各种特征的提取。
"""
import math
import numpy as np
class TrackSequence:
    """
    属性：
    (1)tracksequence：一段轨道序列;【dtype】:np.array 注意:len(tracksequence)>=m+1
    (2)b:轨道取值划分的区间个数，也是计算bin熵时使用的参数 【dtype】:R
    (3)m:序列平移窗口内片段Xi的长度，也是计算tracksequence的近似熵和样本熵时所使用的参数m 【dtype】:R
    (4)r: 两个片段Xi和Xj之间距离的阈值，也是计算tracksequence的近似熵和样本熵时所使用的参数r【dtype】:R
    (5)w:分段聚合逼近产生的新序列的长度【dtype】:R
    方法：
    (1)maximum():tracksequence的最大值;
    (2)minimum():tracksequence的最小值;
    (3)meanvalue()：tracksequence的均值;
    (4)variance():tracksequence的方差(有偏估计);
    (5)skewness():tracksequence的偏度；(注意：偏度可能小于0)
    (6)kurtosis():tracksequence的峰度；
    (7)binnedentropy():tracksequence的bin熵；
    (8)approximateentropy():tracksequence的近似熵(注意：近似熵有可能小于零)
    (9)samlpeentropy():tracksequence的样本熵；
    (10)piecewiseagrregate():tracksequence的分段聚合逼近；
    """
    tracksequence = None
    b = 10
    m = 3
    r = None
    w = 6

    def __init__(self, tracksequence):
        self.tracksequence = np.array(tracksequence)
        self.r = 0.2 * self.tracksequence.std()

    def maximum(self):
        return self.tracksequence.max()


    def minimum(self):
        return self.tracksequence.min()

    def meanvalue(self):
        return self.tracksequence.mean()

    def variance(self):
        return self.tracksequence.var()

    def skewness(self):
        mean = self.meanvalue()
        std = self.tracksequence.std()

        #如果序列一直保持不变，那么其标准差为0，在后续结算过程中会出现处以零的情况,所以需要特殊处理。
        if std == 0:
            std = 1
        sum = 0
        for value in self.tracksequence:
            sum = sum + pow(value-mean, 3)/pow(std, 3)

        skew = sum/len(self.tracksequence)

        return skew

    def kurtosis(self):
        mean = self.meanvalue()
        std = self.tracksequence.std()
        if std == 0:
            std = 1
        sum = 0
        for value in self.tracksequence:
            sum = sum + pow(value - mean, 4) / pow(std, 4)

        kurtosis = sum / len(self.tracksequence)

        return kurtosis


    def binnedentropy(self):

        #分布在每个bin区间的轨道个数
        binprob = [0] * (self.b + 1)
        binwidth = (self.maximum() - self.minimum())/self.b
        mintrack = self.minimum()

        for track in self.tracksequence:
            relativedis = track - mintrack
            if binwidth != 0:
                binid = math.ceil(relativedis/binwidth)
                binprob[binid] += 1


        #区间熵
        binentropy = 0

        for prob in binprob:
            prob = prob/len(self.tracksequence)
            if prob:
                binentropy -= prob * math.log(prob, 10)

        return binentropy


    def approximateentropy(self):

        entropy_madd1 = self.middle_entropy(self.m + 1, 'approximate')
        entropy_m = self.middle_entropy(self.m, 'approximate')
        approximateentropy = entropy_m - entropy_madd1

        return approximateentropy

    def sampleentropy(self):
        A = self.middle_entropy(self.m + 1, 'sample')
        B = self.middle_entropy(self.m, 'sample')

        e = math.exp(1)

        sampleentropy = 0 - math.log(A / B, e)

        return sampleentropy

    #计算approxiamteentropy和sampleentropy的中间量
    def middle_entropy(self, m, entropy_type):
        """
        功能描述：计算approximateetropy 和sampleentropy所需的中间量
        :param m: 片段的长度
        :param entropy_type:熵的类型 【dtype】string
        :return:
        """

        N = len(self.tracksequence)

        # 序列中长度为m的序列片段
        X = [self.tracksequence[i:i + m] for i in range(0, N - m + 1)]

        # 和片段X[i]的距离小于r的片段的个数,至少为1，因为包括本身
        Xi_similarity = [1] * len(X)

        for i in range(0, len(X)):
            for j in range(0, len(X)):
                sub = X[j] - X[i]
                sub = np.array([pow(value, 2) for value in sub])
                dis = math.sqrt(sub.sum())
                if dis < self.r:
                    Xi_similarity[i] += 1

        entropy_m = None
        e = math.exp(1)
        if entropy_type == 'approximate':
            Xi_similarity = np.array([math.log(sim / (N - m + 1), e) for sim in Xi_similarity ])
            entropy_m = Xi_similarity.mean()
        else:
            entropy_m = np.array(Xi_similarity).sum()/2

        return entropy_m


    def piecewiseagrregate(self):

        N = len(self.tracksequence)

        #使用分段聚合逼近产生的新的序列
        new_sequence = [0] * self.w

        for i in range(0, self.w):
            sum = 0
            for j in range(int((N/self.w) * i), int((N/self.w) * (i + 1))):
                sum += self.tracksequence[j]

            new_sequence[i] = sum * (self.w /N)

        return new_sequence

    #将轨道序列的所有特征组成一个向量
    def feature_vector(self):
        vector = []
        vector.append(self.minimum())
        vector.append(self.maximum())
        vector.append(self.meanvalue())
        vector.append(self.variance())
        vector.append(self.skewness())
        vector.append(self.kurtosis())
        vector.append(self.binnedentropy())
        vector.append(self.approximateentropy())
        vector.append(self.sampleentropy())
        vector.extend(self.piecewiseagrregate())
        return vector

#======================================================================
import jieba
from sklearn.feature_extraction.text import CountVectorizer
import pickle

class CorpusOperator:
    """
    描述：与语料有关的一些操作。
    """

    app_function_path = os.path.join(os.getcwd(),'info','app_function.csv')
    dictionary_path = os.path.join(os.getcwd(),'info','dictionary.dict')
    cut_app_function_path = os.path.join(os.getcwd(), 'info', 'cut_app_function.csv')

    def dictionary_build(self):
        """
        描述：读取app_function文件，
        对每一行中的功能文本进行分词；
        将app名称、周数、分词结果一行行写进cut_app_function.csv文件
        根据分词结果建立的二维语料库；
        使用语料库生成字典；
        保存字典。
        :return:
        """

        copus = []
        cut_after = []

        cut_before = read_csv(self.app_function_path, 2)
        for line in cut_before:
            new_line = line[:2]
            function_cut = jieba.cut(''.join(line[2:]))
            function_words = [word for word in function_cut]
            new_line.extend(function_words)

            cut_after.append(new_line)
            copus.append(function_words)

        write_csv(self.cut_app_function_path, cut_after, 'w+')

        copus = [" ".join(doc) for doc in copus]
        countvector = CountVectorizer()
        countvector.fit(copus)

        pickle.dump(countvector.vocabulary_, open(self.dictionary_path, 'wb+'))


if __name__ == '__main__':
    cp = CorpusOperator()
    cp.dictionary_build()
    dic = pickle.load(open('info/dictionary.dict', 'rb'))
    print(len(dic))



