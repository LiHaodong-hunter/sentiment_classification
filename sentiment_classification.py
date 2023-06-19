# 机 构：中国科学院大学
# 程序员：李浩东
# 时 间：2023/6/10 13:23

import os
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as Data
import utils
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix

# 定义超参数
vocab_size = 57078
embedding_dim = 50
hidden_dim = 256
batch_size = 20  # 每次批处理的数据数量
lr = 1e-3  # 学习率
EPOCHS = 10  # 训练样本训练轮数
EPOCH = 1  # 验证、测试样本测试轮数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trainpath = 'Dataset\\train.txt'
validatepath = 'Dataset\\validation.txt'
testpath = 'Dataset\\test.txt'
word2vec_pretrained = 'Dataset\\wiki_word2vec_50.bin'

# 准备数据
word2id, id2word = utils.build_word2id(trainpath, validatepath, testpath)
# print(word2id)
# print(id2word)
word2vecs = utils.build_word2vec(word2vec_pretrained, word2id, save_to_path=None)
#训练数据
traindata = utils.ProcessDataset(trainpath,word2id,id2word)
trainloader = Data.DataLoader(traindata, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=utils.process_data)
#验证数据
validatedata = utils.ProcessDataset(validatepath, word2id, id2word)
validateloader = Data.DataLoader(validatedata, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=utils.process_data)
#测试数据
testdata = utils.ProcessDataset(testpath, word2id, id2word)
testloader = Data.DataLoader(testdata, batch_size=batch_size, shuffle=False, num_workers=2, collate_fn=utils.process_data)


# 创建模型
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, filters_num, filter_size, pre_weight):
        super(TextCNN, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)  # 创建词嵌入层
        self.embeddings.weight.requires_grad = False  # # 固定词嵌入层的参数
        self.embeddings = self.embeddings.from_pretrained(pre_weight)  # 使用预训练的词向量初始化
        self.convs = nn.ModuleList([nn.Conv2d(1, filters_num, (size, embedding_dim)) for size in filter_size])  # 创建多个不同大小的卷积核
        self.dropout = nn.Dropout(0.2)  # 创建一个dropout层
        self.fc1 = nn.Linear(filters_num * len(filter_size), 128)  # 创建两个全连接层，类别数为2
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):  # x的size为(batch_size, max_len)
        x = self.embeddings(x)  #(batch_size, max_len, embedding_dim),将输入数据映射到词嵌入空间中
        x = x.unsqueeze(1)      #(batch_size, 1, max_len, embedding_dim),将数据在卷积之后需要的维度上扩充一维
        x = torch.tensor(x, dtype=torch.float32)  # # 将tensor改为float型Tensor
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # 对每个不同大小的卷积核进行卷积操作，并使用ReLU激活函数
        x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x]  # 对每个卷积结果进行最大池化操作并压缩维度
        x = torch.cat(x, 1)  # 将所有卷积核操作的结果拼接在一起
        x = self.dropout(x)
        out = torch.tanh(self.fc1(x))
        out = self.dropout(out)
        out = self.fc2(out)
        return out


textcnnmodel = TextCNN(vocab_size, embedding_dim, filters_num=128, filter_size=[2,3,5,7,9], pre_weight=word2vecs)  # 创建模型对象

criterion = nn.CrossEntropyLoss()  # 损失函数
optimizer = optim.Adam(textcnnmodel.parameters(), lr=lr, weight_decay=1e-3)  # 优化器

# 训练模型
def train(model, dataset, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        train_correct = 0.0
        train_total = 0.0
        for i, (datas, labels) in enumerate(dataset):
            output = model(datas)  # 前向传播
            # print(output)
            # print(output.data)
            loss = criterion(output, labels)  # 计算损失
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_predict = torch.max(output.data, 1)[1]  # 是计算模型中每个类别的最大值并返回其索引值，即该类别的标签值
            train_correct += (train_predict.to(device) == labels.to(device)).sum()
            train_total += labels.size(0)
            accuracy = train_correct / train_total * 100.0
            print("Train--->Epoch:%d,  Batch:%3d,  Loss:%.8f,  train_correct:%d,  train_total:%d,  accuracy:%.6f" % (
                epoch + 1, i + 1, loss.item(), train_correct, train_total, accuracy))
        validate(textcnnmodel, validateloader, EPOCH)  # 验证


# 验证模型
def validate(model, validateloader, epochs):
    model.eval()
    for epoch in range(epochs):
        val_correct = 0.0
        val_total = 0.0
        loss = 0.0
        for i, (datas, labels) in enumerate(validateloader):
            with torch.no_grad():
                output = model(datas)  # 前向传播
                loss = criterion(output, labels)  # 计算损失
                val_predict = torch.max(output.data, 1)[1]
                val_correct += (val_predict.to(device) == labels.to(device)).sum()
                val_total += labels.size(0)
                accuracy = val_correct / val_total * 100.0
    model.train()
    print("Validate--->Loss:%.8f,  validate_correct:%d,  validate_total:%d,  accuracy:%.6f" % (loss, val_correct, val_total, accuracy))


# 测试模型
def test(model, dataset, criterion, epochs):
    model.eval()
    for epoch in range(epochs):
        labels_lst = []
        test_predict_lst = []
        for i, (datas, labels) in enumerate(dataset):
            with torch.no_grad():
                output = model(datas)  # 前向传播
                # loss = criterion(output, labels)  # 计算损失
                test_predict = torch.max(output.data, 1)[1]
                labels_lst.extend(labels.tolist())  # 使用.tolist()方法将tensor转为list
                test_predict_lst.extend(test_predict.tolist())
        # 计算准确率
        accuracy = accuracy_score(labels_lst, test_predict_lst)
        # 计算召回率
        recall = recall_score(labels_lst, test_predict_lst)
        # 计算F1-分数
        f1 = f1_score(labels_lst, test_predict_lst)
        # 计算混淆矩阵
        confusion = confusion_matrix(labels_lst, test_predict_lst)
        confusion = [[int(x) for x in row] for row in confusion]  # 混淆矩阵中的每个数值都需要转换成int类型
        print("Test--->Accuracy: %.6f, Recall: %.6f, F1 score: %.6f, Confusion matrix:%s" % (accuracy, recall, f1, confusion))


def save_param(model, path):
    torch.save(model.state_dict(), path)  # 保存网络里的参数


def load_param(model, path):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))

if __name__ == "__main__":
    print('-------------------------------------------训练开始-----------------------------------------------')
    # train(textcnnmodel, trainloader, criterion, optimizer, EPOCHS)  # 训练
    #
    # save_param(textcnnmodel, 'textcnnmodel.pth')
    load_param(textcnnmodel, 'textcnnmodel.pth')

    print('-------------------------------------------测试开始-----------------------------------------------')
    test(textcnnmodel, testloader, criterion, EPOCH)  # 测试