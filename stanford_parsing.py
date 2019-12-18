


import os
import json
import corenlp
# os.getenv("CORENLP_HOME") # 환경변수가 잘 지정되었는지 확인하는 코드
text = q
with corenlp.CoreNLPClient(annotators="tokenize ssplit pos lemma ner depparse".split()) as client:
    ann = client.annotate(text)
sentence = ann.sentence[0]
assert corenlp.to_text(sentence) == text
[t.word for t in sentence.token]
[t.lemma for t in sentence.token]

if __name__ == "__main__":
    # read train question, header file
    type = "train"
    path = os.path.join(os.path.dirname(os.getcwd()), "WikiSQL/data/" + type + ".jsonl")
    table_path = os.path.join(os.path.dirname(os.getcwd()), "WikiSQL/data/" + type + ".tables.jsonl")

    qs = open(path).readlines()
    ts = open(table_path).readlines()


    with open(os.path.join(os.path.dirname(os.getcwd()), "QA_wikisql/" + type + ".txt"), "w", encoding="utf-8") as w:
        for line in qs:
            w.write(json.loads(line)["question"])
            w.write("\n")
        for line in ts:
            for word in json.loads(line)["header"]:
                w.write(word)
                w.write("\n")

