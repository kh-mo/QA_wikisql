#-*- coding:utf-8 -*-

import os
import json
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
            infile = open(os.path.join(os.getcwd(), "preprocess/" + type + "_bs_" + str(args.merges) + ".jsonl"), 'r', encoding='utf-8')
            outfile = open(os.path.join(os.getcwd(), "preprocess/" + type + "_bs_" + str(args.merges) + "_restore.jsonl"), 'w', encoding='utf-8')
        else:
            infile = open(os.path.join(os.getcwd(), "preprocess/" + type + "_bs.jsonl"), 'r', encoding='utf-8')
            outfile = open(os.path.join(os.getcwd(), "preprocess/" + type + "_bs_restore.jsonl"), 'w', encoding='utf-8')

        for line in infile:
            doc = json.loads(line)
            if args.use_bpe == "True":
                doc['nli_restore'] = recover_tokenizer(recover_bpe(doc['nli_bs']))
                doc['col_restore'] = []
                for word in doc['col_bs']:
                    doc['col_restore'].append(recover_tokenizer(recover_bpe(word)))
            else:
                doc['nli_restore'] = recover_tokenizer(recover_bpe(doc['nli_b']))
                doc['col_restore'] = []
                for word in doc['col_b']:
                    doc['col_restore'].append(recover_tokenizer(recover_bpe(word)))

            outfile.write(json.dumps(doc, ensure_ascii=False))
            outfile.write("\n")
