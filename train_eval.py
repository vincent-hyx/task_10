# coding: UTF-8
import os

from torch import optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time

from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from focalloss import FocalLoss
from utils import get_time_dif
from pytorch_pretrained_bert.optimization import BertAdam
import more_itertools


scl = False  # if True -> scl + cross entropy loss. else just cross entropy loss
tem = 0.3  # temprature for contrastive loss
lam = 0.9  # lambda for loss
gamma = 1
uda_confidence_thresh = 0.6

import pickle

def save_report(para, my_dict):
    with open("reports/report_" + para+".pkl", "wb") as tf:
        pickle.dump(my_dict, tf)
    # 读取文件
    # my_dict = {'Apple': 4, 'Banana': 2, 'Orange': 6, 'Grapes': 11}
    # with open("myDictionary.pkl", "rb") as tf:
    #    new_dict = pickle.load(tf)


def KL(input, target, reduction="batchmean"):
    target = target.view(-1, 1)
    loss = F.kl_div(F.log_softmax(input, dim=-1, dtype=torch.float32).gather(1, target),
                    F.softmax(target, dim=-1, dtype=torch.float32), reduction=reduction)
    return loss


def contrastive_loss(temp, embedding, label):
    """calculate the contrastive loss
    """
    # cosine similarity between embeddings
    cosine_sim = cosine_similarity(embedding, embedding)
    # remove diagonal elements from matrix
    dis = cosine_sim[~np.eye(cosine_sim.shape[0], dtype=bool)].reshape(cosine_sim.shape[0], -1)
    # apply temprature to elements
    dis = dis / temp
    cosine_sim = cosine_sim / temp
    # apply exp to elements
    dis = np.exp(dis)
    cosine_sim = np.exp(cosine_sim)

    # calculate row sum
    row_sum = []
    for i in range(len(embedding)):
        row_sum.append(sum(dis[i]))
    # calculate outer sum
    contrastive_loss = 0
    for i in range(len(embedding)):
        n_i = label.tolist().count(label[i]) - 1
        inner_sum = 0
        # calculate inner sum
        for j in range(len(embedding)):
            if label[i] == label[j] and i != j:
                inner_sum = inner_sum + np.log(cosine_sim[i][j] / row_sum[i])
        if n_i != 0:
            contrastive_loss += (inner_sum / (-n_i))
        else:
            contrastive_loss += 0
    return contrastive_loss


# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if len(w.size()) < 2:
                continue
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def _get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


def torch_device_one():
    return torch.tensor(1.).to(_get_device())


def get_tsa_thresh(schedule, global_step, num_train_steps, start, end):
    training_progress = torch.tensor(float(global_step) / float(num_train_steps))
    if schedule == 'linear_schedule':
        threshold = training_progress
    elif schedule == 'exp_schedule':
        scale = 5
        threshold = torch.exp((training_progress - 1) * scale)
    elif schedule == 'log_schedule':
        scale = 5
        threshold = 1 - torch.exp((-training_progress) * scale)
    output = threshold * (end - start) + start
    return output.to(_get_device())


def train(config, model, train_iter, dev_iter, test_iter, uda_iter, uda=False):
    start_time = time.time()
    model.train()
    sup_iter = [t for t in enumerate(train_iter)]
    sup_iter_len = len(sup_iter)
    if uda:
        total_step = more_itertools.ilen(uda_iter)
    else:
        total_step = sup_iter_len
    # optimizer = optim.SGD(model.parameters(), lr=config.learning_rate)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=config.learning_rate,
                         warmup=0.05,
                         t_total=total_step * config.num_epochs)

    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    FL = FocalLoss(gamma=gamma)
    uda_softmax_temp = 1
    global_step = 0
    for epoch in range(config.num_epochs):
        iter_bar = tqdm(uda_iter) if uda else tqdm(train_iter)
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        for i, (unsup_train, unsup_labels) in enumerate(iter_bar):

            # print(len(sup_iter))
            j = i % sup_iter_len
            # print(i, j)
            t, (train, labels) = sup_iter[j]
            # print(train)

            outputs = model(train)
            # loss = KL(outputs, labels)
            # loss_sup = FL(outputs, labels)

            # supervised learning part
            """
            if uda:
                # tsa_thresh = get_tsa_thresh('linear_schedule', global_step, total_step, start=1. / outputs.shape[-1], end=1)
                loss_sup = F.cross_entropy(outputs, labels, reduction='none')
                larger_than_threshold = torch.exp(-loss_sup) > 1  # prob = exp(log_prob), prob > tsa_threshold
                # larger_than_threshold = torch.sum(  F.softmax(pred[:sup_size]) * torch.eye(num_labels)[sup_label_ids]  , dim=-1) > tsa_threshold

                loss_mask = torch.ones_like(labels, dtype=torch.float32) * (
                        1 - larger_than_threshold.type(torch.float32))
                loss_mask = torch.detach(loss_mask)
                loss_sup = torch.sum(loss_sup * loss_mask, dim=-1) / torch.max(torch.sum(loss_mask, dim=-1),
                                                                               torch_device_one())
            else:
            """
            # loss_sup = F.cross_entropy(outputs, labels)
            loss_sup = FL(outputs, labels)

            # unsupervised learning part
            if uda:
                logits = model(unsup_train)
                # print(outputs.shape)
                outputs_1 = torch.reshape(logits, [-1, 4])
                # print(outputs_1.shape)
                logp_x = torch.log_softmax(outputs_1[:, :2], dim=-1)
                # logp_x = torch.log_softmax(outputs_1[:, :2], dim=-1)
                # logp_x = torch.detach(logp_x)
                # print(logp_x.shape)
                # p_y = torch.softmax(outputs_1[:, 2:], dim=-1)
                tgt_p_y = torch.softmax(outputs_1[:, 2:], dim=-1)
                tgt_p_y = torch.detach(tgt_p_y)
                loss_unsup = F.kl_div(logp_x, tgt_p_y, reduction='batchmean')
                """
                unsup_loss_mask = torch.max(torch.exp(logp_x), dim=-1)[0] > uda_confidence_thresh
                unsup_loss_mask = torch.detach(unsup_loss_mask)
                unsup_loss_mask = unsup_loss_mask.type(torch.float32)
                # print(p_y.shape)
                loss_unsup = torch.sum(F.kl_div(logp_x, tgt_p_y, reduction='none'), dim=-1)
                loss_unsup = torch.sum(loss_unsup * unsup_loss_mask, dim=-1) / (outputs_1.shape[0] / 2)
                """
            else:
                loss_unsup = 0


            # loss = F.cross_entropy(outputs, labels)
            model.zero_grad()
            loss = loss_sup + loss_unsup
            print(loss_sup, loss_unsup)
            loss.backward()
            optimizer.step()

            if global_step % 10 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    # torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                model.train()
            total_batch += 1

            # 将 train accuracy 保存到 "tensorboard/train" 文件夹
            log_dir = os.path.join('tensorboard', 'train')
            train_writer = SummaryWriter(log_dir=log_dir)
            # 将 test accuracy 保存到 "tensorboard/test" 文件夹
            log_dir = os.path.join('tensorboard', 'test')
            test_writer = SummaryWriter(log_dir=log_dir)

            # 绘制
            train_writer.add_scalar('Loss', loss.item(), global_step)
            test_writer.add_scalar('Loss', dev_loss, global_step)
            global_step += 1
            # if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                # print("No optimization for a long time, auto-stopping...")
                # flag = True
                # break
        # if flag:
            # break
    final_save_path = 'data/saved_dict/bert_uda_final.ckpt' if uda else 'data/saved_dict/bert_baseline_final.ckpt'
    torch.save(model.state_dict(), final_save_path)
    print("\n测试集\n")
    test(config, model, test_iter)


def test(config, model, test_iter):
    # test
    # model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    j = 0
    i = 0
    FL = FocalLoss(gamma=gamma)
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            # outputs = outputs.softmax(dim=1)
            if scl == True:
                # kl_loss = KL(outputs, labels)
                # focalloss = FL(outputs, labels)
                cross_loss = F.cross_entropy(outputs, labels)
                # contrastive_l = contrastive_loss(tem, hidden_state_embedding.cpu().detach().numpy(), labels)
                # loss = (lam * contrastive_l) + (1 - lam) * (cross_loss)
            if scl == False:
                # loss = KL(outputs, labels)
                # loss = FL(outputs, labels)
                loss = F.cross_entropy(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
    for t in range(len(labels_all)):
        if labels_all[t] == 1:
            i += 1
            if predict_all[t] == 1:
                j += 1
    print(f"\nsexist number:{i}, acc in sexist samples:{j/i}\n")
    acc = metrics.accuracy_score(labels_all, predict_all)
    if test:
        report = metrics.classification_report(labels_all, predict_all,
                                               target_names=config.class_list,
                                               digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion
    return acc, loss_total / len(data_iter)
