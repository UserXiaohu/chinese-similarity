from text2vec import Similarity
import jieba
import heapq
import jieba.analyse
import pandas as pd


# 初始化获取停用词表
stop = open('stop_word.txt', 'r+', encoding='utf-8')
stopword = stop.read().split("\n")
key = open('key_word.txt', 'r+', encoding='utf-8')
keyword = key.read().split("\n")

"""
加载初始数据信息
str:文件传输路径
index:所需真实值索引列表
"""


def init_data(str, index):
    dream_data = pd.read_csv(str)
    return dream_data.values[:, index]


"""
对文本内容进行过滤
1。过滤停用词
2。结合关键词/字过滤
"""


def strip_word(seg):
    # 打开写入关键词的文件
    jieba.load_userdict("./key_word.txt")
    # print("去停用词：\n")
    wordlist = []
    # 获取关键字
    keywords = jieba.analyse.extract_tags(seg, topK=5, withWeight=False, allowPOS=('n'))
    # 遍历分词表
    for key in jieba.cut(seg):
        # print(key)
        # 去除停用词，去除单字且不在关键词库，去除重复词
        if not (key.strip() in stopword) and (len(key.strip()) > 1 or key.strip() in keyword) and not (
                key.strip() in wordlist):
            wordlist.append(key)
            # print(key)

    # 停用词去除END
    stop.close()
    return ''.join(wordlist)


"""
通过text2vec词向量模型计算
出来两段处理后的文本相似度
"""


def similarity_calculation(str_arr, str_2):
    sim = Similarity()
    str_2 = strip_word(str_2)
    result = []
    for item in str_arr:
        #这里可以将base提前处理好导出备用，以达到优化目的
        item = strip_word(item)
        result.append(sim.get_score(item, str_2))
    return result


"""
将用户细节文本描述
转换为关键词文本
"""


def deal_init_data(text_data):
    text_arr = []
    for item in text_data:
        # 做关键词提取
        text_arr.append(strip_word(item))
    key_words = pd.DataFrame(text_arr, columns=['key_text'])
    key_words.to_csv('dream_keywords.csv', sep=',', header=True, index=True)
    return key_words


if __name__ == '__main__':
    # 读取文本的对比数据关键词
    key_arr = init_data('base_content.csv', 1);
    # 读取文本的对比数据关键词
    demo_arr = init_data('demo.csv', 0);

    for index,item in enumerate(demo_arr):
        result = similarity_calculation(key_arr,item)
        # 获取相似度最高的前三个
        re1 = map(result.index, heapq.nlargest(3, result))
        re2 = heapq.nlargest(3, result)
        print("原文本",item)
        for i, val in enumerate(list(re1)):
            print(i+1,".对比结果：",key_arr[val],"，相似度：",re2[i])
