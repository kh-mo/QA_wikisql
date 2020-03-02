import os
import json
import argparse
from collections import defaultdict

def count_voca(dataset, args):
    voca = defaultdict(int)
    length = []
    for sent in dataset:
        doc = json.loads(sent)
        if args.use_bpe == "True":
            words = (doc['nl_sb'] + ' ' + ' '.join(doc['col_sb'])).split()
        else:
            words = (doc['nl_s'] + ' ' + ' '.join(doc['col_s'])).split()
        length.append(len(words))
        for word in words:
            voca[word] += 1
    mean_length = sum(length) / len(length)
    return voca, mean_length

def check_voca(dataset, check_list, args):
    voca = defaultdict(int)
    for sent in dataset:
        doc = json.loads(sent)
        if args.use_bpe == "True":
            words = (doc['nl_sb'] + ' ' + ' '.join(doc['col_sb'])).split()
        else:
            words = (doc['nl_s'] + ' ' + ' '.join(doc['col_s'])).split()
        for word in words:
            if word not in check_list:
                voca[word] += 1
    return voca

def save_voca(dataset, args):
    if args.merges == -1:
        with open(os.path.join(os.getcwd(), "preprocess/bpe_util_data/voca_none.txt"), "w", encoding='utf-8') as w:
            for word in dataset:
                w.write(word)
                w.write("\n")
    else:
        with open(os.path.join(os.getcwd(), "preprocess/bpe_util_data/voca_"+str(args.merges)+".txt"), "w", encoding='utf-8') as w:
            for word in dataset:
                w.write(word)
                w.write("\n")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--merges', type=int, default=-1)
    args = parser.parse_args()

    if args.merges == -1:
        train_data = open(os.path.join(os.getcwd(), "preprocess/stanford/train_s.jsonl"), 'r', encoding='utf-8').readlines()
        dev_data = open(os.path.join(os.getcwd(), "preprocess/stanford/dev_s.jsonl"), 'r', encoding='utf-8').readlines()
        test_data = open(os.path.join(os.getcwd(), "preprocess/stanford/test_s.jsonl"), 'r', encoding='utf-8').readlines()
        args.use_bpe = "False"
    else:
        train_data = open(os.path.join(os.getcwd(), "preprocess/bpe/train_sb_" + str(args.merges) + ".jsonl"), 'r', encoding='utf-8').readlines()
        dev_data = open(os.path.join(os.getcwd(), "preprocess/bpe/dev_sb_" + str(args.merges) + ".jsonl"), 'r', encoding='utf-8').readlines()
        test_data = open(os.path.join(os.getcwd(), "preprocess/bpe/test_sb_" + str(args.merges) + ".jsonl"), 'r', encoding='utf-8').readlines()
        args.use_bpe = "True"

    ## trainset에 있는 모든 unique 단어들을 get
    train_voca, train_mean_length = count_voca(train_data, args)
    train_voca_list = sorted(list(train_voca.keys()))

    ## dev, testset에 있는 단어들 중 trainset에 없는 unique 단어 get
    dev_voca = check_voca(dev_data, train_voca_list, args)
    test_voca = check_voca(test_data, train_voca_list, args)

    dev_oov, test_oov = list(dev_voca.keys()), list(test_voca.keys())

    print("Train voca 수 : {}개".format(len(train_voca_list)))
    print("Train sentence 평균 길이 : {}".format(train_mean_length))
    print("Dev OOV 수 : {}개".format(len(dev_oov)))
    print("Test OOV 수 : {}개".format(len(test_oov)))

    ## 사전 추출
    save_voca(train_voca_list, args)