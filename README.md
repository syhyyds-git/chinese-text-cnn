# chinese-text-cnn
基于TextCNN模型的文本分类程序示例

## 一、程序运行所需环境
- Python 3.5

- pytorch 1.9.0

- numpy 最新版即可

- scikit-learn 最新版即可

## 二.程序的介绍

Data文件夹下的cnews文件夹中存储的是程序所用数据集和（Word2Vec）模型文件。
cnews.train,cnews.val,cnews.test
分别对应着训练，验证，测试
sgns_clean.char（Word2Vec）模型文件，用于词嵌入。


run_model文件包含全部代码，主要有以下几个代码块:
- 配置参数
- 加载数据
- 定义Textcnn模型
- 训练模型
- 测试和预测
- 分类报告和混合矩阵


