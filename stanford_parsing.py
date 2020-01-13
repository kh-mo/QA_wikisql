#-*- coding:utf-8 -*-

import os
import json
import corenlp
# os.getenv("CORENLP_HOME") # 환경변수가 잘 지정되었는지 확인하는 코드

def replace_origin_preprocessing(word):
    # 공백 대체 : replace(u'\xa0', u' ')
    # 큰 따옴표가 stanford parser에서 원하는대로 변환되지 않아 작은따옴표 두개로 대체 : replace("\"", "\'\'")
    return word.replace(u'\xa0', u' ').replace("\"", "``").strip()

def replace_tokenizing_preprocessing(word):
    # 공백 대체 : replace(u'\xa0', u' ')
    # 큰 따옴표가 stanford parser에서 원하는대로 변환되지 않아 작은따옴표 두개로 대체 : replace("\"", "\'\'")
    # 본래 공백을 __로 대체 : replace(" ", "__")
    return word.replace(u'\xa0', u' ').replace("\"", "``").replace(" ", "__")

def write_tokening_preprocessing(word):
    # parsing 이후 "__ "을 "__"로 대체하여 본래 띄어쓰기가 있었던 지점을 표현한다(추후 이것을 이용해 원래 문장으로 복원한다)
    return word.replace("__ ", "__").replace("--", "-").strip()

if __name__ == "__main__":
    # read type question, header file
    types = ["train", "dev", "test"]

    for type in types:
        path = os.path.join(os.path.dirname(os.getcwd()), "WikiSQL/data/" + type + ".jsonl")
        table_path = os.path.join(os.path.dirname(os.getcwd()), "WikiSQL/data/" + type + ".tables.jsonl")

        qs = open(path).readlines() # question sentence
        ts = open(table_path).readlines() # table sentence

        # corenlp parser
        client = corenlp.CoreNLPClient(annotators="tokenize ssplit pos lemma ner depparse".split())

        # make preprocess folder
        preprocessed_dir = os.path.join(os.getcwd(), "preprocess")
        if not os.path.isdir(preprocessed_dir): # 폴더 없으면 생성 있으면 패스
            os.mkdir(preprocessed_dir)

        # original data
        with open(os.path.join(preprocessed_dir, type + ".txt"), "w", encoding="utf-8") as w:
            try:
                for line in qs:
                    w.write(replace_origin_preprocessing(json.loads(line)["question"]))
                    w.write("\n")
                for line in ts:
                    for word in json.loads(line)["header"]:
                        w.write(replace_origin_preprocessing(word))
                        w.write("\n")
            except IndexError as e:
                pass
        print("get {} original done.".format(type))

        # tokenizing
        with open(os.path.join(preprocessed_dir, type + "_token.txt"), "w", encoding="utf-8") as w:
            try:
                for line in qs:
                    ann = client.annotate(replace_tokenizing_preprocessing(json.loads(line)["question"]))
                    # corenlp로 파싱된 question 단어들을 space로 분할하여 한 문장으로 저장
                    w.write(write_tokening_preprocessing(" ".join([t.word for sent in ann.sentence for t in sent.token])))
                    w.write("\n")
                for line in ts:
                    for word in json.loads(line)["header"]:
                        ann = client.annotate(replace_tokenizing_preprocessing(word))
                        # corenlp로 파싱된 header 단어들을 space로 분할하여 한 문장으로 저장
                        w.write(write_tokening_preprocessing(" ".join([t.word for sent in ann.sentence for t in sent.token])))
                        w.write("\n")
            except IndexError as e:
                pass
        print("word {} tokenizing done.".format(type))

        with open(os.path.join(preprocessed_dir, type + "_lemma.txt"), "w", encoding="utf-8") as w:
            try:
                for line in qs:
                    ann = client.annotate(replace_tokenizing_preprocessing(json.loads(line)["question"]))
                    # corenlp로 파싱된 question 단어들을 space로 분할하여 lemmatization 문장으로 저장
                    w.write(write_tokening_preprocessing(" ".join([t.lemma for sent in ann.sentence for t in sent.token])))
                    w.write("\n")
                for line in ts:
                    for word in json.loads(line)["header"]:
                        ann = client.annotate(replace_tokenizing_preprocessing(word))
                        # corenlp로 파싱된 header 단어들을 space로 분할하여 lemmatization 문장으로 저장
                        w.write(write_tokening_preprocessing(" ".join([t.lemma for sent in ann.sentence for t in sent.token])))
                        w.write("\n")
            except IndexError as e:
                pass
        print("word {} lemmatization done.".format(type))


