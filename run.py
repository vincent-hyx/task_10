# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif
from exlanation_analysis import given_explanation
# parser = argparse.ArgumentParser(description='Chinese Text Classification')
# parser.add_argument('--model', type=str, required=True, help='choose a model: Bert, ERNIE')
# args = parser.parse_args()
from eval_dev_task_a import eval_dev

# run baseline then run uda via loading baseline
load_model = False
baseline = False
uda = True

if __name__ == '__main__':
    dataset = 'data'  # 数据集

    model_name = 'bert'  # bert
    x = import_module('models.' + model_name)
    config = x.Config(dataset)
    if uda:
        config.save_path = "data/saved_dict/bert_uda.ckpt"
        config.num_epochs = 3
        config.learning_rate = 3e-5
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    # torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    train_data = build_dataset(config, path=config.train_path, uda=False)
    dev = build_dataset(config, path=config.dev_path, uda=False)
    test = build_dataset(config, path=config.test_path, uda=False)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev, config)
    test_iter = build_iterator(test, config)

    config.batch_size = config.uda_batch_size

    if baseline:
        uda_path = 'data/sub_train.txt'
    else:
        uda_path = 'data/aug_all_train.txt'
    uda_train = build_dataset(config, path=uda_path, uda=True)
    uda_iter = build_iterator(uda_train, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    model = x.Model(config).to(config.device)
    if load_model:
        model.load_state_dict(torch.load('data/saved_dict/bert_baseline.ckpt'))
    train(config, model, train_iter, dev_iter, test_iter, uda_iter, uda=uda)
