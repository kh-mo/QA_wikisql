import numpy as np
import pandas as pd

def loading_data(data_path):
    # R에서 title과 contents만 csv로 저장한걸 불러와서 제목과 컨텐츠로 분리
    # write.csv(corpus, data_path, fileEncoding='utf-8', row.names=F)
    corpus = np.array(pd.read_table(data_path, sep=",", encoding="utf-8"))
    title = []
    body = []
    for idx, doc in enumerate(corpus):
        title.append(doc[0].split())
        body.append(doc[1].split())
        if idx % 100000 == 0:
            print('%d docs' % (idx))
    return np.asarray(title), np.asarray(body)

def convert_dic(input):
    uniq_word = []
    for i in range(len(input)):
        uniq_word = list(set(uniq_word+input[i]))
    uniq_word.sort()
    idx = list(range(len(uniq_word)))
    word_to_idx = dict(zip(uniq_word, idx))
    idx_to_word = dict(zip(idx, uniq_word))
    return word_to_idx, idx_to_word

def get_word_idx(batch, word_to_idx):
    batch_list = []
    for sentence in batch:
        sentence_list = []
        for word in sentence:
            sentence_list.append(word_to_idx.get(word))
        batch_list.append(sentence_list)
    return batch_list