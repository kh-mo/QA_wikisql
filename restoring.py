#-*- coding:utf-8 -*-

import os
import argparse

def set_hyperparameters():
    parser = argparse.ArgumentParser()

    # input file
    parser.add_argument("--infile", default=os.path.join(os.getcwd(), "preprocess/test_bpe.txt"))

    # output file
    parser.add_argument("--outfile", default=os.path.join(os.getcwd(), "preprocess/restore_test_bpe.txt"))
    args = parser.parse_args()
    return args

def recover_bpe(line):
    sent = line.replace("@@ ", '')
    return sent

def recover_tokenizer(line):
    sent = line.replace(" ","").replace("__"," ").strip()
    if "-LRB-" in sent:
        sent = sent.replace("-LRB-", "(")
    if "-RRB-" in sent:
        sent = sent.replace("-RRB-", ")")
    return sent

if __name__ == "__main__":
    args = set_hyperparameters()
    types = ["train", "dev", "test"]

    for type in types:
        infile = open(os.path.join(os.getcwd(), "preprocess/"+type+"_token.txt"), 'r', encoding='utf-8')
        outfile = open(os.path.join(os.getcwd(), "preprocess/restore_"+type+"_token.txt"), 'w', encoding='utf-8')

        for line in infile:
            outfile.write(recover_tokenizer(line))
            outfile.write("\n")
            # outfile.write(recover_tokenizer(recover_bpe(line)))

    # infile = open(args.infile, 'r', encoding='utf-8')
    # outfile = open(args.outfile, 'w', encoding='utf-8')
