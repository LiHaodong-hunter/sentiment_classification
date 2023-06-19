# 机 构：中国科学院大学
# 程序员：李浩东
# 时 间：2023/6/10 20:46

import torch
from torch.utils.data import Dataset
from zhconv import convert
import gensim
import numpy as np

# 词到id的映射(word2id)和id到词的映射(id2word)。
def build_word2id(trainpath, validatepath, testpath):
    word2id = {'_PAD_': 0}
    id2word = {0: '_PAD_'}
    paths = [trainpath, validatepath, testpath]
    for path in paths:
        with open(path, encoding='utf-8') as f:
            for line in f.readlines():
                line = convert(line, 'zh-cn')  # 将文本转换为简体中文
                words = line.strip().split()
                for word in words[1:]:
                    if word not in word2id.keys():
                        word2id[word] = len(word2id)
    for key, val in word2id.items():
        id2word[val] = key
    return word2id, id2word

def build_word2vec(file, word2id, save_to_path=None):
    model = gensim.models.KeyedVectors.load_word2vec_format(file, binary=True)  # # 获取预训练好的词向量模型，该模型是用gensim库得到的二进制文件
    # 生成一个大小为 [n_words, model.vector_size] 的二维矩阵，其中每一行代表一个词的词向量
    n_words = max(word2id.values()) + 1
    word_vecs = np.array(np.random.uniform(-1., 1., [n_words, model.vector_size]))
    # 用预训练好的词向量替代词向量表中的随机向量
    for word in word2id.keys():
        try:
            word_vecs[word2id[word]] = model[word]
        except KeyError:
            pass  # # 如果预训练的词向量库中不存在某个词，则跳过它
    # 如果指定了保存路径，则将构建好的词向量表写入文件
    if save_to_path:
        with open(save_to_path, 'w', encoding='utf-8') as f:
            for vec in word_vecs:
                vec = [str(w) for w in vec]
                f.write(' '.join(vec))
                f.write('\n')
    word2vecs = torch.from_numpy(word_vecs)  # # 将构建好的词向量表转化为PyTorch张量
    return word2vecs


class ProcessDataset(Dataset):
    def __init__(self, file, word2id, id2word):
        self.file = file
        self.word2id = word2id
        self.id2word = id2word
        self.datas, self.labels = self.separate()

    def __getitem__(self, index):
        return self.datas[index], self.labels[index]

    def __len__(self):
        return len(self.datas)

    def separate(self):
        datas, labels = [], []
        with open(self.file, encoding='utf-8') as f:
            for line in f.readlines():
                #取每行的label
                label = torch.tensor(int(line[0]), dtype=torch.int64)
                labels.append(label)
                #取每行的word
                line = convert(line, 'zh-cn')
                line_words = line.strip().split()[1:-1]
                indexs = []
                for word in line_words:
                    try:
                        index = self.word2id[word]
                    except BaseException:
                        index = 0
                    indexs.append(index)
                datas.append(indexs)
            return datas, labels


def process_data(data):
    # 分离data、label
    data.sort(key=lambda x: len(x[0]), reverse=True)  # 对于输入的数据进行排序，按照长度从大到小排序
    input_data = []
    label_data = []
    for i in data:
        input_data.append(i[0])
        label_data.append(i[1])

    # 对于每个输入数据，将其长度截断或补齐到长度为75，大于75截断、小于75补0
    padded_datas = []
    for data in input_data:
        if len(data) >= 75:
            padded_data = data[:75]
        else:
            padded_data = data
            while (len(padded_data) < 75):
                padded_data.append(0)
        padded_datas.append(padded_data)

    # 将标签数据和整理好的输入数据转化为tensor类型
    label_data = torch.tensor(label_data)
    padded_datas = torch.tensor(padded_datas, dtype=torch.int64)
    return padded_datas, label_data  # # 返回整理好的输入数据和标签数据

