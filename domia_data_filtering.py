import numpy as np
import torch
from tqdm import tqdm

from models.bert import Config, Model
from utils import load_dataset, build_iterator


def filtering(data_path, bs_model, save_path):
    config = Config("data")
    config.batch_size = 64
    data = load_dataset(config, data_path , config.pad_size, exist_label=False)
    data_iter = build_iterator(data, config)
    models = Model(config).to(config.device)
    models.load_state_dict(torch.load(bs_model))
    models.eval()
    filter_all = np.array([], dtype=bool)
    with torch.no_grad():
        for text, label in tqdm(data_iter):
            out = models(text)
            pro = torch.softmax(out, dim=-1)
            pro_max = torch.max(pro, 1)[0].cpu().numpy()
            filter_list = pro_max > 0.95
            filter_all = np.append(filter_all, filter_list)
    write_all = []
    with open(data_path, 'r', encoding="UTF-8") as f:
        all = f.readlines()
        i = 0
        for item in filter_all:
            if item:
                write_all.append(all[i])
            i += 1
    with open(save_path, 'w', encoding="UTF-8") as f:
        f.writelines(write_all)
        print(len(write_all))


if __name__ == "__main__":
    data_path = 'data/gab_1M_unlabelled.txt'
    bs_model = 'data/saved_dict/bert_baseline.ckpt'
    save_path = 'data/filter.txt'
    filtering(data_path, bs_model, save_path)






