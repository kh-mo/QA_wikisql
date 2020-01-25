#-*- coding:utf-8 -*-

import os
import argparse

def set_hyperparameters():
    parser = argparse.ArgumentParser()
    parser.add_argument('--merges', type=int, default=-1)
    parser.add_argument('--use_bpe', type=str, default="False")
    args = parser.parse_args()
    return args

def recover_bpe(line):
    sent = line.replace(" ", "").replace("@@", '').replace("__", " ")
    return sent

def recover_tokenizer(line):
    sent = line.strip()
    sent = sent.replace("-LRB-", "(").replace("-RRB-", ")")
    sent = sent.replace("-LSB-", "[").replace("-RSB-", "]")
    sent = sent.replace("-LCB-", "{").replace("-RCB-", "}")
    sent = sent.replace("Jr..", "Jr.").replace("Seq..", "Seq.")
    sent = sent.replace("etc..", "etc.").replace("Etc..", "Etc..").replace("est..","est.").replace("Est..", "Est.")
    sent = sent.replace("Jan..", "Jan.").replace("Feb..", "Feb.").replace("Mar..", "Mar.").replace("Apr..", "Apr.")
    sent = sent.replace("Jun..", "Jun.").replace("Jul..", "Jul.").replace("Aug..", "Aug.").replace("Sep..", "Sep.")
    sent = sent.replace("Oct..", "Oct.").replace("Nov..", "Nov.").replace("Dec..", "Dec.")
    return sent

if __name__ == "__main__":
    args = set_hyperparameters()
    types = ["train", "dev", "test"]

    for type in types:
        if args.use_bpe == "True":
            infile = open(os.path.join(os.getcwd(), "preprocess/" + type + "_bpe_" + str(args.merges) + ".txt"), 'r', encoding='utf-8')
            outfile = open(os.path.join(os.getcwd(), "preprocess/" + type + "_restore_bpe_" + str(args.merges) + ".txt"), 'w', encoding='utf-8')
        else:
            infile = open(os.path.join(os.getcwd(), "preprocess/" + type + "_token_basic.txt"), 'r', encoding='utf-8')
            outfile = open(os.path.join(os.getcwd(), "preprocess/" + type + "_restore_token_basic.txt"), 'w', encoding='utf-8')

        for line in infile:
            outfile.write(recover_tokenizer(recover_bpe(line)))
            outfile.write("\n")
            # outfile.write(recover_tokenizer(recover_bpe(line)))

    # infile = open(args.infile, 'r', encoding='utf-8')
    # outfile = open(args.outfile, 'w', encoding='utf-8')
