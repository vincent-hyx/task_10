# coding: UTF-8
import torch
from tqdm import tqdm
import time
from datetime import timedelta
import csv
import numpy as np
from word_level_augment import clean_english_text_from_nltk

PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号
addtoken = False

def shuffle(data):
    data = np.array(data, dtype=object)
    index = [i for i in range(len(data))]
    np.random.shuffle(index)
    shuffle_data = data[index]
    return shuffle_data


def data_split(data_path):
    contents = []
    with open(data_path, 'r', encoding='UTF-8') as f:
        all = f.readlines()
        for line in tqdm(all):
            # lin = line.strip()
            contents.append(line)
    np.random.seed(42)
    contents = shuffle(contents)
    print(len(contents))
    sub_train = contents[:int(0.9 * len(contents))]
    sub_dev = contents[int(0.90 * len(contents)):int(0.95 * len(contents))]
    sub_test = contents[int(0.95 * len(contents)):]
    print(len(sub_train))
    with open('data/sub_train.txt', 'w', encoding='UTF-8') as f:
        f.writelines(sub_train)
    with open('data/sub_dev.txt', 'w', encoding='UTF-8') as f:
        f.writelines(sub_dev)
    with open('data/sub_test.txt', 'w', encoding='UTF-8') as f:
        f.writelines(sub_test)

# 分割数据集
# data_split('data/train.txt')



def load_dataset(config, path, pad_size=64, exist_label=True):
    contents = []
    with open(path, 'r', encoding='UTF-8') as f:
        all = f.readlines()
        for line in tqdm(all):
            lin = line.strip()
            if exist_label:
                content, label_str = lin.split('\t')
            else:
                content = lin
            if exist_label:
                if label_str == 'not sexist':
                    label = 0
                else:
                    label = 1
            else:
                label = 0
            content = clean_english_text_from_nltk(content)
            token = config.tokenizer.tokenize(content)
            # if config.model_name == 'roberta':
            token = [CLS] + token
            seq_len = len(token)
            mask = []
            token_ids = config.tokenizer.convert_tokens_to_ids(token)
            # token_ids = [config.tokenizer.cls_token_id] + token_ids

            if pad_size:
                if seq_len < pad_size:
                    mask = [1] * seq_len + [0] * (pad_size - seq_len)
                    token_ids += ([0] * (pad_size - seq_len))
                else:
                    mask = [1] * pad_size
                    token_ids = token_ids[:pad_size]
                    seq_len = pad_size
            contents.append([token_ids, label, seq_len, mask])
    return contents



def build_dataset(config, path, uda=False):
    if uda:
        return load_dataset(config, path, config.pad_size, exist_label=False)
    else:
        return load_dataset(config, path, config.pad_size, exist_label=True)

    # train_uda = shuffle(load_dataset(unsup_path, config.pad_size, exist_label=False))
    # dev = load_dataset(config.dev_path, config.pad_size)
    # test = load_dataset(config.test_path, config.pad_size)



class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        return (x, seq_len, mask), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def texts_pre_process(texts, config, pad_size=64, label=1):
    content = []
    for piece in texts:
        token = config.tokenizer.tokenize(piece)
        seq_len = len(token) + 1
        mask = []
        token_ids = config.tokenizer.convert_tokens_to_ids(token)
        token_ids = [config.tokenizer.cls_token_id] + token_ids
        if pad_size:
            if seq_len < pad_size:
                mask = [1] * seq_len + [0] * (pad_size - seq_len)
                token_ids += ([0] * (pad_size - seq_len))
            else:
                mask = [1] * pad_size
                token_ids = token_ids[:pad_size]
                seq_len = pad_size
        content.append([token_ids, label, seq_len, mask])
    if len(texts) > 1:
        return DatasetIterater(content, 100, config.device)
    else:
        return DatasetIterater(content, 1, config.device)

