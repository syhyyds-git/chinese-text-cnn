# @title参数配置
# ================================
#  TextCNN 中文新闻分类 (PyTorch)
# 数据集：THUCNews（cnews）
# 复现任务
# ================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
# !pip install gensim
import jieba, numpy as np, os, gensim
jieba.initialize()
# 配置参数
class Config:
    vocab_size = 5000 #词表大小
    embed_dim = 300 #嵌入矩阵词向量维度
    num_classes = 10 #分类种类
    num_filters = 256
    kernel_sizes = [3, 4, 5]
    dropout = 0.5
    lr = 1e-3
    num_epochs = 10
    batch_size = 64
    max_len = 600
    hidden_dim=256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    word2vec_path = "E:\\data\\sgns_clean.char"
config = Config()
print("Using device:", config.device)



# @title加载数据
from collections import Counter
"""
读取数据并形成数据迭代器
"""
base_dir='E:\\data\\cnews'
train_path=os.path.join(base_dir,'cnews.train.txt')
val_path=os.path.join(base_dir,'cnews.val.txt')
test_path=os.path.join(base_dir,'cnews.test.txt')
#读文件，返回contents(已分词),labels
def read_file(filename):
  contents,labels=[],[]
  with open(filename,'r',encoding='utf-8') as f:
    for line in f:
      parts=line.strip().split('\t',1)#1表示只分割一次
      if len(parts)!=2:
        continue
      label,content=parts[0],parts[1]
      contents.append(list(jieba.lcut(content)))
      labels.append(label)
  return contents,labels

#构建词表
def build_vocab(contents,vocab_size=5000):
  vocabs=[word for content in contents for word in content]
  counter=Counter(vocabs)
  count_most_vocabs=counter.most_common(vocab_size-2)
  most_vocabs={"<PAD>":0,"<UNK>":1}
  for ids,(vocab,_) in enumerate(count_most_vocabs):
    most_vocabs[vocab]=ids+2
  return most_vocabs

#将文本转换成数字序列
def encode_contents(contents,vocab,max_len=600):
  ids=[]
  for content in contents:
    seq=[vocab.get(w,1) for w in content[:max_len]]
    seq+=[0]*(max_len-len(seq))
    ids.append(seq)
  return torch.LongTensor(ids)

categories=['体育','财经','房产','家居','教育','科技','时尚','时政','游戏','娱乐']
cat2id={cat:ids for ids,cat in enumerate(categories)}

print("正在加载词表...")
train_data,train_labels=read_file(train_path)
vocab=build_vocab(train_data,config.vocab_size)
#加载数据
def load_dataset(path,vocab,cat2id,max_len=600):
  contents,labels=read_file(path)
  ids=encode_contents(contents,vocab,max_len)
  labels=torch.LongTensor([cat2id[label] for label in labels])
  return ids,labels

x_train,y_train=load_dataset(train_path,vocab,cat2id,config.max_len)
x_val,y_val=load_dataset(val_path,vocab,cat2id,config.max_len)
x_test,y_test=load_dataset(test_path,vocab,cat2id,config.max_len)

#用 TensorDataset 把输入（x）和标签（y）打包成一个可索引的数据集对象。
train_loader = DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=config.batch_size, shuffle=True,num_workers=0)
val_loader   = DataLoader(torch.utils.data.TensorDataset(x_val, y_val), batch_size=config.batch_size,num_workers=0)
test_loader  = DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=config.batch_size,num_workers=0)
#num_workers=,防止0jieba 的线程模型和 PyTorch 的 DataLoader 多进程机制冲突 导致的死锁

print(f"数据加载完成：训练 {len(x_train)}，验证 {len(x_val)}，测试 {len(x_test)}")



# @title定义Textcnn模型
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gensim
import os

class TextCNN(nn.Module):
    def __init__(self, config, vocab, word2vec_path):
        super(TextCNN, self).__init__()

        
        self.embedding = nn.Embedding(config.vocab_size, config.embed_dim)

        
        if config.word2vec_path is not None and os.path.exists(config.word2vec_path):
            print("正在加载 Word2Vec 预训练向量...")
            w2v_model = gensim.models.KeyedVectors.load_word2vec_format(
                config.word2vec_path, binary=False
            )
            print(f"Word2Vec 向量维度: {w2v_model.vector_size}")

            # 初始化随机矩阵
            embedding_matrix = np.random.uniform(
                -0.25, 0.25, (config.vocab_size, config.embed_dim)
            ).astype(np.float32)
            ##NumPy 的均匀分布随机数生成函数
            # 替换为预训练词向量
            found = 0
            for word, idx in vocab.items():
                if word in w2v_model:
                    embedding_matrix[idx] = w2v_model[word]
                    found += 1
            self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
            print(f"Word2Vec 加载完成，匹配到 {found}/{len(vocab)} 个词向量。")
        else:
            print("未加载 Word2Vec 预训练向量，使用随机初始化。")
            """
            torch.from_numpy(embedding_matrix):
            将 NumPy 数组 embedding_matrix 转换为 PyTorch 张量
            NumPy 的数据类型通常是 float64，转换成 PyTorch
            后会变成 torch.float32（可训练的浮点数张量）。
            self.embedding.weight 是 Embedding 层的权重张量，
            形状为 (vocab_size, embed_dim)
            .data 表示直接操作底层数据，不会追踪梯度（防止对梯度计算造成干扰）
            copy_ 是 就地复制，把右边的张量内容复制到左边的张量中
            """
        # 卷积层 
        self.convs = nn.ModuleList([
            nn.Conv2d(1, config.num_filters, (k, config.embed_dim))
            for k in config.kernel_sizes
        ])
        """
        ModuleList:PyTorch 提供的容器，用来存放多个子模块
        1:输入通道
        config.num_filters：输出通道，每种k有多少个卷积核
        """

        # ========== Dropout + 全连接层 ==========
        self.dropout = nn.Dropout(config.dropout)
        self.hidden_fc = nn.Linear(config.num_filters * len(config.kernel_sizes),
                                   config.hidden_dim)
        self.output_fc = nn.Linear(config.hidden_dim, config.num_classes)

    # 卷积 + 池化
    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)) # [batch, num_filters, seq_len-k+1, 1]
        x = x.squeeze(3)   # [batch, num_filters, seq_len-k+1]
        x = F.max_pool1d(x, x.size(2)) # [batch, num_filters, 1]
        x = x.squeeze(2) # [batch, num_filters]
        return x

    # 前向传播
    def forward(self, x):
        x = self.embedding(x).unsqueeze(1) # [batch, 1, seq_len, embed_dim]
        """
        在第 1 个位置插入一个新维度。
        这个新维度通常用于表示 通道数 (channel)。
        在TextCNN中,相当于把“文本嵌入矩阵”看成一张单通道的灰度图像。
        """
        out = torch.cat([self.conv_and_pool(x, conv) for conv in self.convs], 1)
        #cat将每种k对应的[bitch_size,config.num_filters]沿着dim=1方向
        #合在一起形成[bitch_size,config.num_filters*len(kernel_sizes)]
        out = self.dropout(out)
        out = F.relu(self.hidden_fc(out))
        out = self.output_fc(out)
        return out




model = TextCNN(config, vocab, config.word2vec_path).to(config.device)
print(model)



# @title训练模型
optimizer=torch.optim.Adam(model.parameters(),lr=config.lr)
criterion=nn.CrossEntropyLoss()

def evaluate(model,data_loader):
  model.eval()
  pred,labels=[],[]
  with torch.no_grad():
    for x_batch,y_batch in data_loader:
      x_batch,y_batch=x_batch.to(config.device),y_batch.to(config.device)
      outputs=model(x_batch)
      pred+=torch.argmax(outputs,1).cpu().tolist()
      labels+=y_batch.cpu().tolist()
    acc=accuracy_score(labels,pred)
    f1=f1_score(labels,pred,average='macro')
    return acc,f1
def train():
  model.train()
  best_f1=0
  for epoch in range(config.num_epochs):
    for x_train,y_train in train_loader:
      x_train,y_train=x_train.to(config.device),y_train.to(config.device)
      outputs=model(x_train)
      optimizer.zero_grad()
      loss=criterion(outputs,y_train)
      loss.backward()
      optimizer.step()
    acc,f1=evaluate(model,val_loader)
    print(f"Epoch {epoch+1}/{config.num_epochs} | Val Acc={acc:.4f} | F1={f1:.4f}")
    if f1>best_f1:
      best_f1=f1
      torch.save(model.state_dict(),"textcnn_best.pth")
      """
      model.state_dict()
      返回模型的所有可训练参数（权重和偏置）的字典（OrderedDict）
      """
      print("模型已保存")
train()



# @title测试和预测
def test():
  state_dict = torch.load("textcnn_best.pth", weights_only=True)
  model.load_state_dict(state_dict)
  model.eval()
  acc,f1=evaluate(model,test_loader)
  print(f"Test Acc={acc:.4f},F1={f1:.4f}")
def predict(text):
  model.eval()
  words = list(jieba.lcut(text))
  seq = [vocab.get(w, 1) for w in words[:config.max_len]]
  seq += [0] * (config.max_len - len(seq))
  seq = torch.LongTensor(seq).unsqueeze(0).to(config.device)
  #unsqueeze(0)增加一层batch_size
  with torch.no_grad():
    output=model(seq)
    pred=torch.argmax(output,1).item()
  print("输入：", text)
  print("预测类别：", categories[pred])
test()
predict("黄蜂vs湖人首发：科比带伤战保罗 加索尔救赎之战")
predict("上周新增基金开户数重回4万")


# @title分类报告和混淆矩阵
model.load_state_dict(torch.load("textcnn_best.pth"))
model.eval()
preds,trues=[],[]
with torch.no_grad():
  for x_batch,y_batch in test_loader:
    x_batch,y_batch=x_batch.to(config.device),y_batch.to(config.device)
    outputs=model(x_batch)
    pred=torch.argmax(outputs,1)
    preds+=pred.cpu().tolist()
    trues+=y_batch.cpu().tolist()
  print("分类报告:")
  print(classification_report(trues,preds,target_names=categories))
  print("混淆矩阵:")
  print(confusion_matrix(trues,preds))





