import csv
import re

import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
from word_level_augment import clean_english_text_from_nltk
from csv2txt import filter_emoji
from exlanation_analysis import load_model
from utils import build_iterator
addtoken = False
PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号


def evaluate(model, data_iter ,test):
    model.eval()
    predict_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            # outputs = outputs.softmax(dim=1)
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            predict_all = np.append(predict_all, predic)
    return predict_all


def build_dataset(config, data_path):

    def shuffle(data):
        data = np.array(data, dtype=object)
        index = [i for i in range(len(data))]
        np.random.shuffle(index)
        shuffle_data = data[index]
        return shuffle_data

    def load_dataset(path, pad_size=64, exist_label=False):
        contents = []
        with open(path, 'r', encoding='UTF-8') as f:
            reader = csv.reader(f)
            for line in tqdm(reader):
                # lin = line.strip()
                if not line:
                    continue
                if reader.line_num == 1:
                    continue
                # content, label = line.split('\t')
                s = str(filter_emoji(line[1]))
                s = re.sub('\[USER\]', '', s)
                content = re.sub('\[URL\]', '', s)

                if exist_label:
                    if line[2] == 'not sexist':
                        label = 0
                    else:
                        label = 1
                else:
                    label = 0
                content = clean_english_text_from_nltk(content)
                print(content)
                token = config.tokenizer.tokenize(content)
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
    train = load_dataset(data_path, config.pad_size, exist_label=False)
    # train_uda = shuffle(load_dataset(unsup_path, config.pad_size, exist_label=False))
    # dev = load_dataset(config.dev_path, config.pad_size)
    # test = load_dataset(config.test_path, config.pad_size)
    return train


def eval_dev():

    model, config = load_model('data', 'bert')
    model.eval()
    test_data = build_dataset(config, config.test_data_path)
    test_iter = build_iterator(test_data, config)
    predict_all = evaluate(model, test_iter, test=True)
    # time_dif = get_time_dif(start_time)
    # print("Time usage:", time_dif)
    data = pd.read_csv('data/test_task_a_entries.csv', names=['rewire_id', 'text'])
    rewire_id = str(data['rewire_id'][0])
    data1 = list(data['rewire_id'])[1:]
    print(type(data1))
    label_pred = 'label_pred'
    data2 = []
    sexist_num = 0
    not_sexist_num = 0
    for i in range(len(predict_all)):
        if predict_all[i] == 1:
            data2.append('sexist')
            sexist_num += 1
        else:
            if predict_all[i] == 0:
                data2.append('not sexist')
                not_sexist_num += 1
            else:
                print("数据异常")
                break
    print(f"sexist number:{sexist_num} || not sexist number:{not_sexist_num}")
    dataframe = pd.DataFrame({rewire_id: data1, label_pred: data2})
    dataframe.to_csv('data/test_result_uda.csv', index=False, sep=',')


"""
def data_filter():
    model, config = load_model('data', 'roberta')
    model.load_state_dict(torch.load(config.baseline_model_path))
    model.eval()
    filter_data = build_dataset(config, config.filter_data_path)
    filter_iter = build_iterator(filter_data, config)
    predict_all = np.array([], dtype=float)
    with torch.no_grad():
        for texts, labels in filter_iter:
            outputs, hidden_state_embedding = model(texts)
            outputs = outputs.softmax(dim=1)
            predic = torch.max(outputs.data, 1)[0].cpu().numpy()
            predict_all = np.append(predict_all, predic)
    with open('data/gab_1M_unlabelled.txt', "r", encoding="UTF-8"):
        for p in predict_all:
            if p >
"""





if __name__ == '__main__':
    eval_dev()
