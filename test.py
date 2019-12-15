## step 1. 파일 읽어들이기
import os
import json

type = "train"
path = os.path.join(os.getcwd(), "data/"+type+".jsonl")
table_path = os.path.join(os.getcwd(), "data/"+type+".tables.jsonl")

qs = open(path).readlines()
ts = open(table_path).readlines()

## step 2. qeustion, table_id, table header에 접근
t_infor = {}
for line in ts:
    doc = json.loads(line)
    t_infor[doc["id"]] = doc["header"]

q = json.loads(qs[0])['question']
q_t = json.loads(qs[0])['table_id']

## step 3. stanford corenlp로 tokenizing
import corenlp
# os.getenv("CORENLP_HOME") # 환경변수가 잘 지정되었는지 확인하는 코드
text = q
with corenlp.CoreNLPClient(annotators="tokenize ssplit pos lemma ner depparse".split()) as client:
    ann = client.annotate(text)
sentence = ann.sentence[0]
assert corenlp.to_text(sentence) == text
[t.word for t in sentence.token]
[t.lemma for t in sentence.token]
'''
lemma를 학습에 사용하고 word를 이용해서 원복하는 코드가 따로 존재해야 할 듯
'''

## step 4. bpe 수행위한 train, test, 데이터 셋 추출하기
with open(os.path.join(os.getcwd(), "train.txt"),"w", encoding="utf-8") as w:
    for line in ts:
        for word in json.loads(line)["header"]:
            w.write(word)
            w.write("\n")
    for line in qs:
        w.write(json.loads(line)["question"])
        w.write("\n")

