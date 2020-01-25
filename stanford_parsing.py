#-*- coding:utf-8 -*-

import os
import json
import corenlp
import requests
# os.getenv("CORENLP_HOME") # 환경변수가 잘 지정되었는지 확인하는 코드

def replace_origin_preprocessing(word):
    fraction_dic = {"¼":"(1/4)", "½":"(1/2)", "¾":"(3/4)", "⅐":"(1/7)", "⅑":"(1/9)",
                    "⅒":"(1/10)", "⅓":"(1/3)", "⅔":"(2/3)", "⅕":"(1/5)", "⅖":"(2/5)",
                    "⅗":"(3/5)", "⅘":"(4/5)", "⅙":"(1/6)", "⅚":"(5/6)", "⅛":"(1/8)",
                    "⅜":"(3/8)", "⅝":"(5/8)", "⅞":"(7/8)", "↉":"(0/3)"}

    for key in list(fraction_dic.keys()):
        if key in word:
            word = word.replace(key, fraction_dic[key])

    # 유니코드 대체, 공백 통일
    word = word.replace(u'\xa0', u' ').replace(u'\u2009', u' ').replace('\t', ' ')
    # 따옴표 통일
    word = word.replace("“","``").replace("”","``").replace("‘","`").replace("’","`").replace("´","`").replace("\"","``").replace("\'","`")
    # 하이픈 통일
    word = word.replace("–", "-").replace("—", "-").replace("——", "-")
    # %문제 해결위해 띄어쓰기(task specific)
    word = word.replace("%2006", "% 2006").replace("%2001", "% 2001").replace("%females", "% females")
    # 생략부호 교체, 좌우 공백 제거
    word = word.replace("…", "...").strip()

    return word

def replace_tokenizing_preprocessing(word):
    # 본래 공백을 __로 대체 : replace(" ", "__")
    return replace_origin_preprocessing(word).replace(" ", "__")

def write_tokening_preprocessing(word):
    # parsing 이후 "__ "을 "__"로 대체하여 본래 띄어쓰기가 있었던 지점을 표현한다(추후 이것을 이용해 원래 문장으로 복원한다)
    return word.replace("__ ", "__").strip()

def make_dir(path):
    # make preprocess folder
    if not os.path.isdir(path):  # 폴더 없으면 생성 있으면 패스
        os.mkdir(path)
        print("complete making directory")
    print("Directory already exist")

def make_origin_data(path):
    # read type question, header file
    types = ["train", "dev", "test"]
    for type in types:
        question_path = os.path.join(os.path.dirname(os.getcwd()), "WikiSQL/data/" + type + ".jsonl")
        table_path = os.path.join(os.path.dirname(os.getcwd()), "WikiSQL/data/" + type + ".tables.jsonl")

        qs = open(question_path).readlines()  # question sentence
        ts = open(table_path).readlines()  # table sentence

        # original data
        with open(os.path.join(path, type + ".txt"), "w", encoding="utf-8") as w:
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

def make_check_idx(path):
    if os.path.isfile(path):
        data = open(path, 'r', encoding='utf-8').readlines()
        check_idx = len(data)
    else:
        check_idx = 0
    return check_idx

def make_tokenizing_data(path):
    # corenlp parser
    client = corenlp.CoreNLPClient(annotators="tokenize ssplit pos lemma ner depparse".split())
    types = ["train", "dev", "test"]
    for type in types:
        origin_data = open(os.path.join(path, type + ".txt"), 'r', encoding='utf-8').readlines()

        # tokenizing
        tokenizing_path = os.path.join(path, type + "_token_basic.txt")
        token_check_idx = make_check_idx(tokenizing_path)
        with open(tokenizing_path, "a", encoding="utf-8") as tw:
            for idx, line in enumerate(origin_data):
                if idx >= token_check_idx:
                    ann = client.annotate(replace_tokenizing_preprocessing(line))
                    # corenlp로 파싱된 단어들을 space로 분할하여 특수토큰 '__'을 추가한 문장으로 저장
                    tw.write(write_tokening_preprocessing(" ".join([t.word for sent in ann.sentence for t in sent.token])))
                    tw.write("\n")
        print("word {} tokenizing done.".format(type))

        # lemmatizing
        lemmatization_path = os.path.join(path, type + "_lemma_basic.txt")
        lemma_check_idx = make_check_idx(lemmatization_path)
        with open(lemmatization_path, "a", encoding="utf-8") as lw:
            for idx, line in enumerate(origin_data):
                if idx >= lemma_check_idx:
                    ann = client.annotate(replace_tokenizing_preprocessing(line))
                    # corenlp로 파싱된 question 단어들을 space로 특수토큰 '__'을 추가한 lemmatization 문장으로 저장
                    lw.write(write_tokening_preprocessing(" ".join([t.lemma for sent in ann.sentence for t in sent.token])))
                    lw.write("\n")
        print("word {} lemmatization done.".format(type))

if __name__ == "__main__":
    preprocessed_dir = os.path.join(os.getcwd(), "preprocess")
    make_dir(preprocessed_dir)
    make_origin_data(preprocessed_dir)
    while True:
        try:
            make_tokenizing_data(preprocessed_dir)
            break
        except requests.exceptions.ConnectionError:
            continue

# import os
# os.chdir("C:\\Users\\mosi\\Desktop\\workspace\\nli/QA_wikisql")
# types="dev"
# origin_data = open(os.path.join(os.getcwd(), "preprocess/"+types+".txt"), 'r', encoding='utf-8').readlines()
# origin_data2 = open(os.path.join(os.getcwd(), "preprocess/"+types+"_token_-1.txt"), 'r', encoding='utf-8').readlines()
# origin_data3 = open(os.path.join(os.getcwd(), "preprocess/"+types+"_lemma_-1.txt"), 'r', encoding='utf-8').readlines()
# print(len(origin_data))
# print(len(origin_data2))
# print(len(origin_data3))
