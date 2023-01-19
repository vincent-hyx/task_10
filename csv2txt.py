import os
import csv
from tqdm import tqdm


def to_txt(csv_path, txt_path):
    content_all = []
    with open(csv_path, 'r', encoding='UTF-8') as f:
        reader = csv.reader(f)
        for line in tqdm(reader):
            # lin = line.strip()
            if not line:
                continue
            if reader.line_num == 1:
                continue
            s = str(filter_emoji(line[1]))
            s = re.sub('\[USER\]', '', s)
            s = re.sub('\[URL\]', '', s)
            content = s + '\n'
            content_all.append(content)
    with open(txt_path, 'w', encoding='UTF-8') as f:
        f.writelines(content_all)

import re
import emoji

def filter_emoji(content):
    p = re.compile(u'['u'\U0001F300-\U0001F64F' u'\U0001F680-\U0001F6FF' u'\u2600-\u2B55 \U00010000-\U0010ffff]+')
    return re.sub(p, u' ', content)


if __name__ == '__main__':
    """
    path = 'data/train.txt'
    i = 0
    with open(path, 'r', encoding='UTF-8') as f:
        all = f.readlines()
        for line in all:
            line = line.strip('\n')
            s = line.split('\t')
            if s[1] == "sexist":
                label = 0
            else:
                if s[1] == "not sexist":
                    label = 1
                else:
                    print("format error")
                    print(s[1])
                    i += 1
    print(i)
    """
    to_txt('data/train.csv', "data/train_unlabelled.txt")
    # to_txt("data/gab_1M_unlabelled.csv", "data/gab_1M_unlabelled.txt")
    # to_txt("data/reddit_1M_unlabelled.csv", "data/reddit_1M_unlabelled.txt")




