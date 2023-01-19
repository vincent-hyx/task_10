# coding: UTF-8
import torch
import torch.nn as nn
# from pytorch_pretrained_bert import BertModel, BertTokenizer
from transformers import BertModel, BertTokenizer


class Config(object):

    """配置参数"""

    def __init__(self, dataset):
        self.model_name = 'bert'
        self.train_path = dataset + '/train.txt'  # 训练集
        self.dev_path = dataset + '/sub_dev.txt'                                    # 验证集
        self.test_path = dataset + '/sub_test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/class.txt').readlines()]  # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name + '_baseline.ckpt'  # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备

        self.require_improvement = 500000  # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)  # 类别数
        self.num_epochs = 8  # epoch数
        self.batch_size = 16  # mini-batch大小
        self.uda_batch_size = 32
        self.pad_size = 90  # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-5  # 学习率
        self.bert_path = './bert_pretrained'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.test_data_path = dataset + '/test_task_a_entries.csv'
        self.uda_train_path = dataset + '/eda_train_unlabelled.txt'
        self.uda_path_gab = dataset + '/eda_gab_1M_unlabelled.txt'
        self.uda_path_reddit = dataset + '/eda_reddit_1M_unlabelled.txt'
        self.save_path_final = dataset + '/saved_dict/' + self.model_name + '_final' + '.ckpt'
        self.save_path_uda = dataset + '/saved_dict/' + self.model_name + '_uda' + '.ckpt'
        self.filter_data_path = 'data/gab_1M_unlabelled.txt'
        self.baseline_model_path = 'data/saved_dict/bert_baseline.ckpt'


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)
        self.dropout = nn.Dropout(p=0.2)
        print(f"config.num_classes:{config.num_classes}")

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        output = self.bert(context, attention_mask=mask)
        pooled = output[1]
        out = self.fc(pooled)
        return out
