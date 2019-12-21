#-*- coding:utf-8 -*-

import os
import json
import corenlp
# os.getenv("CORENLP_HOME") # 환경변수가 잘 지정되었는지 확인하는 코드
# assert corenlp.to_text(sentence) == text
# text = "test document"
# with corenlp.CoreNLPClient(annotators="tokenize ssplit pos lemma ner depparse".split()) as client:
#     ann = client.annotate(text)
# sentence = ann.sentence[0]
# [t.word for t in sentence.token]
# [t.lemma for t in sentence.token]

if __name__ == "__main__":
    # read train question, header file
    type = "train"
    path = os.path.join(os.path.dirname(os.getcwd()), "WikiSQL/data/" + type + ".jsonl")
    table_path = os.path.join(os.path.dirname(os.getcwd()), "WikiSQL/data/" + type + ".tables.jsonl")

    qs = open(path).readlines()
    ts = open(table_path).readlines()

    # corenlp parser
    client = corenlp.CoreNLPClient(annotators="tokenize ssplit pos lemma ner depparse".split())

    # make preprocess folder
    preprocessed_dir = os.path.join(os.getcwd(), "preprocess")
    if not os.path.isdir(preprocessed_dir):#폴더 없으면 생성 있으면 패스
        os.mkdir(preprocessed_dir)

    # tokenizing
    with open(os.path.join(preprocessed_dir, type + ".txt"), "w", encoding="utf-8") as w:
        try:
            for line in qs:
                ann = client.annotate(json.loads(line)["question"])
                # corenlp로 파싱된 question 단어들을 space로 분할하여 한 문장으로 저장
                w.write(" ".join([t.word for t in ann.sentence[0].token]))
                w.write("\n")
            for line in ts:
                for word in json.loads(line)["header"]:
                    ann = client.annotate(word)
                    # corenlp로 파싱된 header 단어들을 space로 분할하여 한 문장으로 저장
                    w.write(" ".join([t.word for t in ann.sentence[0].token]))
                    w.write("\n")
        except IndexError as e:
            pass
    print("word tokenizing done.")

    with open(os.path.join(preprocessed_dir, type + "_lemma.txt"), "w", encoding="utf-8") as w:
        try:
            for line in qs:
                ann = client.annotate(json.loads(line)["question"])
                # corenlp로 파싱된 question 단어들을 space로 분할하여 한 문장으로 저장
                w.write(" ".join([t.lemma for t in ann.sentence[0].token]))
                w.write("\n")
            for line in ts:
                for word in json.loads(line)["header"]:
                    ann = client.annotate(word)
                    # corenlp로 파싱된 header 단어들을 space로 분할하여 한 문장으로 저장
                    w.write(" ".join([t.lemma for t in ann.sentence[0].token]))
                    w.write("\n")
        except IndexError as e:
            pass
    print("lemma tokenizing done.")