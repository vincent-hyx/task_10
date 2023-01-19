import numpy as np

from models import bert, roberta
from importlib import import_module
import torch
from lime.lime_text import LimeTextExplainer
from utils import texts_pre_process
import matplotlib.pyplot as plt


def load_model(dataset, model_name):
    x = import_module('models.' + model_name)
    config = x.Config(dataset)
    model = x.Model(config).to(config.device)
    model.load_state_dict(torch.load('data/saved_dict/bert_uda_final.ckpt'))
    return model, config


def model_predict(texts):
    dataset = 'data'
    model_name = 'roberta'
    model, config = load_model(dataset, model_name)
    print(len(texts))
    data_iter = texts_pre_process(texts, config)
    model.eval()
    predict_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts_1, labels in data_iter:
            outputs, hidden_state_embedding = model(texts_1)
            pre = outputs.softmax(dim=-1).cpu().numpy()
            predict_all = np.append(predict_all, pre)
            preditc = predict_all.reshape(-1, 2)
    return preditc


def given_explanation(texts, label, class_names, num_features):
    explainer = LimeTextExplainer(class_names=class_names)
    print(texts)
    print(class_names)
    exp = explainer.explain_instance(texts, model_predict, num_features=num_features)
    print('Probability(sexist) =', model_predict([texts])[0, 1])
    print('True class: %s' % class_names[label])
    exp.as_list()

    print(exp)
    fig = exp.as_pyplot_figure()
    exp.show_in_notebook(text=False)
    # exp.save_to_file('/tmp/oi.html')
    exp.show_in_notebook(text=True)
    plt.show()


if __name__ == '__main__':
    class_names = list(['not sexist', 'sexist'])
    text = 'When women\'s delusions are confronted with reality, women double down on delusions.'
    given_explanation(text, label=0, class_names=class_names, num_features=6)
