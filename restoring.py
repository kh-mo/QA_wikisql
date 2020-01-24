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
    sent = line.replace(" ", "").replace("__", " ").strip()
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
        infile = open(os.path.join(os.getcwd(), "preprocess/"+type+"_token.txt"), 'r', encoding='utf-8')
        outfile = open(os.path.join(os.getcwd(), "preprocess/restore_"+type+"_token.txt"), 'w', encoding='utf-8')

        for line in infile:
            outfile.write(recover_tokenizer(line))
            outfile.write("\n")
            # outfile.write(recover_tokenizer(recover_bpe(line)))

    # infile = open(args.infile, 'r', encoding='utf-8')
    # outfile = open(args.outfile, 'w', encoding='utf-8')
