# coding=utf-8
# Copyright 2019 The Google UDA Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Word level augmentations including Replace words with uniform random words or TF-IDF based word replacement.
"""


import collections
import copy
import json
import math
import re
import string
import random

import nltk

'''
from absl import flags
import numpy as np
import tensorflow as tf

FLAGS = flags.FLAGS

printable = set(string.printable)


def build_vocab(examples):
    vocab = {}

    def add_to_vocab(word_list):
        for word in word_list:
            if word not in vocab:
                vocab[word] = len(vocab)

    for i in range(len(examples)):
        add_to_vocab(examples[i].word_list_a)
    return vocab


def filter_unicode(st):
    return "".join([c for c in st if c in printable])


class EfficientRandomGen(object):
    """A base class that generate multiple random numbers at the same time."""

    def reset_random_prob(self):
        """Generate many random numbers at the same time and cache them."""
        cache_len = 100000
        self.random_prob_cache = np.random.random(size=(cache_len,))
        self.random_prob_ptr = cache_len - 1

    def get_random_prob(self):
        """Get a random number."""
        value = self.random_prob_cache[self.random_prob_ptr]
        self.random_prob_ptr -= 1
        if self.random_prob_ptr == -1:
            self.reset_random_prob()
        return value

    def get_random_token(self):
        """Get a random token."""
        token = self.token_list[self.token_ptr]
        self.token_ptr -= 1
        if self.token_ptr == -1:
            self.reset_token_list()
        return token


class UnifRep(EfficientRandomGen):
    """Uniformly replace word with random words in the vocab."""

    def __init__(self, token_prob, vocab):
        self.token_prob = token_prob
        self.vocab_size = len(vocab)
        self.vocab = vocab
        self.reset_token_list()
        self.reset_random_prob()

    def __call__(self, example):
        example.word_list_a = self.replace_tokens(example.word_list_a)
        if example.text_b:
            example.word_list_b = self.replace_tokens(example.word_list_b)
        return example

    def replace_tokens(self, tokens):
        """Replace tokens randomly."""
        if len(tokens) >= 3:
            if np.random.random() < 0.001:
                show_example = True
            else:
                show_example = False
            if show_example:
                tf.logging.info("before augment: {:s}".format(
                    filter_unicode(" ".join(tokens))))
            for i in range(len(tokens)):
                if self.get_random_prob() < self.token_prob:
                    tokens[i] = self.get_random_token()
            if show_example:
                tf.logging.info("after augment: {:s}".format(
                    filter_unicode(" ".join(tokens))))
        return tokens

    def reset_token_list(self):
        """Generate many random tokens at the same time and cache them."""
        self.token_list = self.vocab.keys()
        self.token_ptr = len(self.token_list) - 1
        np.random.shuffle(self.token_list)


def get_data_stats(examples):
    """Compute the IDF score for each word. Then compute the TF-IDF score."""
    word_doc_freq = collections.defaultdict(int)
    # Compute IDF
    for i in range(len(examples)):
        cur_word_dict = {}
        cur_sent = copy.deepcopy(examples[i].word_list_a)
        if examples[i].text_b:
            cur_sent += examples[i].word_list_b
        for word in cur_sent:
            cur_word_dict[word] = 1
        for word in cur_word_dict:
            word_doc_freq[word] += 1
    idf = {}
    for word in word_doc_freq:
        idf[word] = math.log(len(examples) * 1. / word_doc_freq[word])
    # Compute TF-IDF
    tf_idf = {}
    for i in range(len(examples)):
        cur_word_dict = {}
        cur_sent = copy.deepcopy(examples[i].word_list_a)
        if examples[i].text_b:
            cur_sent += examples[i].word_list_b
        for word in cur_sent:
            if word not in tf_idf:
                tf_idf[word] = 0
            tf_idf[word] += 1. / len(cur_sent) * idf[word]
    return {
        "idf": idf,
        "tf_idf": tf_idf,
    }


class TfIdfWordRep(EfficientRandomGen):
    """TF-IDF Based Word Replacement."""

    def __init__(self, token_prob, data_stats):
        super(TfIdfWordRep, self).__init__()
        self.token_prob = token_prob
        self.data_stats = data_stats
        self.idf = data_stats["idf"]
        self.tf_idf = data_stats["tf_idf"]
        data_stats = copy.deepcopy(data_stats)
        tf_idf_items = data_stats["tf_idf"].items()
        tf_idf_items = sorted(tf_idf_items, key=lambda item: -item[1])
        self.tf_idf_keys = []
        self.tf_idf_values = []
        for key, value in tf_idf_items:
            self.tf_idf_keys += [key]
            self.tf_idf_values += [value]
        self.normalized_tf_idf = np.array(self.tf_idf_values)
        self.normalized_tf_idf = (self.normalized_tf_idf.max()
                                  - self.normalized_tf_idf)
        self.normalized_tf_idf = (self.normalized_tf_idf
                                  / self.normalized_tf_idf.sum())
        self.reset_token_list()
        self.reset_random_prob()

    def get_replace_prob(self, all_words):
        """Compute the probability of replacing tokens in a sentence."""
        cur_tf_idf = collections.defaultdict(int)
        for word in all_words:
            cur_tf_idf[word] += 1. / len(all_words) * self.idf[word]
        replace_prob = []
        for word in all_words:
            replace_prob += [cur_tf_idf[word]]
        replace_prob = np.array(replace_prob)
        replace_prob = np.max(replace_prob) - replace_prob
        replace_prob = (replace_prob / replace_prob.sum() *
                        self.token_prob * len(all_words))
        return replace_prob

    def __call__(self, example):
        if self.get_random_prob() < 0.001:
            show_example = True
        else:
            show_example = False
        all_words = copy.deepcopy(example.word_list_a)
        if example.text_b:
            all_words += example.word_list_b

        if show_example:
            tf.logging.info("before tf_idf_unif aug: {:s}".format(
                filter_unicode(" ".join(all_words))))

        replace_prob = self.get_replace_prob(all_words)
        example.word_list_a = self.replace_tokens(
            example.word_list_a,
            replace_prob[:len(example.word_list_a)]
        )
        if example.text_b:
            example.word_list_b = self.replace_tokens(
                example.word_list_b,
                replace_prob[len(example.word_list_a):]
            )

        if show_example:
            all_words = copy.deepcopy(example.word_list_a)
            if example.text_b:
                all_words += example.word_list_b
            tf.logging.info("after tf_idf_unif aug: {:s}".format(
                filter_unicode(" ".join(all_words))))
        return example

    def replace_tokens(self, word_list, replace_prob):
        """Replace tokens in a sentence."""
        for i in range(len(word_list)):
            if self.get_random_prob() < replace_prob[i]:
                word_list[i] = self.get_random_token()
        return word_list

    def reset_token_list(self):
        cache_len = len(self.tf_idf_keys)
        token_list_idx = np.random.choice(
            cache_len, (cache_len,), p=self.normalized_tf_idf)
        self.token_list = []
        for idx in token_list_idx:
            self.token_list += [self.tf_idf_keys[idx]]
        self.token_ptr = len(self.token_list) - 1
        tf.logging.info("sampled token list: {:s}".format(
            filter_unicode(" ".join(self.token_list))))


def word_level_augment(
        examples, aug_ops, vocab, data_stats):
    """Word level augmentations. Used before augmentation."""
    if aug_ops:
        if aug_ops.startswith("unif"):
            tf.logging.info("\n>>Using augmentation {}".format(aug_ops))
            token_prob = float(aug_ops.split("-")[1])
            op = UnifRep(token_prob, vocab)
            for i in range(len(examples)):
                examples[i] = op(examples[i])
        elif aug_ops.startswith("tf_idf"):
            tf.logging.info("\n>>Using augmentation {}".format(aug_ops))
            token_prob = float(aug_ops.split("-")[1])
            op = TfIdfWordRep(token_prob, data_stats)
            for i in range(len(examples)):
                examples[i] = op(examples[i])
    return examples
'''


from nltk.corpus import wordnet
from bs4 import BeautifulSoup

def clean_english_text_from_nltk(text):
    """
    使用nltk的停用词对英文数据进行清洗
    :param text:
    :return:
    """
    text = BeautifulSoup(text,'html.parser').get_text() #去掉html标签
    text = re.sub(r"can\'t", "can not", text)
    text = re.sub(r"cannot", "can not ", text)
    text = re.sub(r"what\'s", "what is", text)
    text = re.sub(r"What\'s", "what is", text)
    text = re.sub(r"\'ve ", " have ", text)
    text = re.sub(r"n\'t", " not ", text)
    text = re.sub(r"n’t", " not ", text)
    text = re.sub(r"i\'m", "i am ", text)
    text = re.sub(r"I\'m", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"‘re", " are ", text)
    text = re.sub(r"\'s", " is ", text)
    text = re.sub(r"’s", " is ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r" e mail ", " email ", text)
    text = re.sub(r" e \- mail ", " email ", text)
    text = re.sub(r" e\-mail ", " email ", text)
    text = re.sub(r'[^a-zA-Z]',' ',text) #除去标点符号
    words = text.lower().split() #转为小写并切分
    # stopwords = nltk.corpus.stopwords.words('english') #使用nltk的停用词
    # wordList =[word for word in words if word not in stopwords]

    return ' '.join(words)

stop_words = ['i', 'me', 'my', 'myself', 'we', 'our',
              'ours', 'ourselves', 'you', 'your', 'yours',
              'yourself', 'yourselves', 'he', 'him', 'his',
              'himself', 'she', 'her', 'hers', 'herself',
              'it', 'its', 'itself', 'they', 'them', 'their',
              'theirs', 'themselves', 'what', 'which', 'who',
              'whom', 'this', 'that', 'these', 'those', 'am',
              'is', 'are', 'was', 'were', 'be', 'been', 'being',
              'have', 'has', 'had', 'having', 'do', 'does', 'did',
              'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or',
              'because', 'as', 'until', 'while', 'of', 'at',
              'by', 'for', 'with', 'about', 'against', 'between',
              'into', 'through', 'during', 'before', 'after',
              'above', 'below', 'to', 'from', 'up', 'down', 'in',
              'out', 'on', 'off', 'over', 'under', 'again',
              'further', 'then', 'once', 'here', 'there', 'when',
              'where', 'why', 'how', 'all', 'any', 'both', 'each',
              'few', 'more', 'most', 'other', 'some', 'such', 'no',
              'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
              'very', 's', 't', 'can', 'will', 'just', 'don',
              'should', 'now', 're', 'ain', 'th']


def synonym_replacement(words, words_tfidf, threshold):
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in stop_words and (\
        words_tfidf.get(word) is not None and words_tfidf[word] < threshold)]))
    random.seed(42)
    random.shuffle(random_word_list)
    num_replaced = 0
    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(list(synonyms))
            new_words = [synonym if word == random_word else word for word in new_words]
            # print("replaced", random_word, "with", synonym)
            num_replaced += 1
            if num_replaced >= 4:  # only replace up to n words
                break
    # this is stupid but we need it, trust me
    sentence = ' '.join(new_words)
    new_words = sentence.split(' ')

    return new_words


def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonym = l.name().replace("_", " ").replace("-", " ").lower()
            synonym = "".join([char for char in synonym if char in ' qwertyuiopasdfghjklzxcvbnm'])
            synonyms.add(synonym)
    if word in synonyms:
        synonyms.remove(word)
    return list(synonyms)

def get_only_chars(line):

    clean_line = ""
    # re.sub(r'[^a-zA-Z]', ' ', text)  # 除去标点符号
    # line = re.sub('you\'re', 'you are', line)
    # line = re.sub('you\'re','you are', line)
    # line = re.sub('ain\'t','are not', line)
    line = line.replace("’", "")
    line = line.replace("'", "")
    line = line.replace("-", " ") #replace hyphens with spaces
    line = line.replace("\t", " ")
    line = line.replace("\n", " ")
    line = line.lower()


    for char in line:
        if char in 'qwertyuiopasdfghjklzxcvbnm ':
            clean_line += char
        else:
            clean_line += ' '

    clean_line = re.sub(' +',' ',clean_line) #delete extra spaces
    if clean_line[0] == ' ':
        clean_line = clean_line[1:]
    return clean_line


if __name__ == '__main__':
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
    # 语料库路径
    path = 'data/train_unlabelled.txt'
    texts = []
    with open(path, 'r', encoding="UTF-8") as f:
        lines = f.readlines()
        for line in lines:
            lin = line.split('\t')
            clean = clean_english_text_from_nltk(lin[0].strip())
            texts.append(clean)
    # texts = ["hello world same is is is", "hello vincent model", "hello anna task"]
    # a = ["hello world", "hello vincent", "hello anna"]
    vectorizer = CountVectorizer(max_features=None)
    # 计算每一个词语的TF-IDF权值
    tf_idf_transformer = TfidfTransformer()
    # 计算每一个词语出现的次数#将文本转换为词频并计算tf-idf;fit_transform()方法用于计算每一个词语出现的次数
    X = vectorizer.fit_transform(texts)
    y = vectorizer.get_feature_names()
    # print(X)
    # print(y)
    '''
    ['anna', 'hello', 'vincent', 'world']
    [[0.         0.50854232 0.         0.861037  ]
    [0.         0.50854232 0.861037   0.        ]
    [0.861037   0.50854232 0.         0.        ]]
    '''
    tf_idf = tf_idf_transformer.fit_transform(X)

    # tf_idf_matrices = tf_idf.toarray()
    i = 0

    with open('data/aug_all_train.txt', 'w', encoding="UTF-8") as f:
        aug_texts = []
        for text in texts:
            words = text.split(' ')
            words = [word for word in words if word != '']
            # print(tf_idf_matrices[i])
            words_tfidf = {}
            word_idf_len = len(tf_idf[i].indices)
            for j in range(word_idf_len):
                words_tfidf[y[tf_idf[i].indices[j]]] = tf_idf[i].data[j]
            # print(text)
            # print(words)
            #  print(words_tfidf)
            aug_texts.append(' '.join(synonym_replacement(words, words_tfidf, 0.9)) + '\n')
            aug_texts.append(text + '\n')
            i += 1
        f.writelines(aug_texts)




