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

def recover(line):
    sent = line.replace("@@ ", '')
    return sent

if __name__ == "__main__":
    args = set_hyperparameters()
    infile = open(args.infile, 'r', encoding='utf-8')
    outfile = open(args.outfile, 'w', encoding='utf-8')
    infile = open(os.path.join(os.getcwd(), "preprocess/test_bpe.txt"), 'r', encoding='utf-8')
    outfile = open(os.path.join(os.getcwd(), "preprocess/restore_test_bpe.txt"), 'w', encoding='utf-8')
    for line in infile:
        outfile.write(recover(line))
