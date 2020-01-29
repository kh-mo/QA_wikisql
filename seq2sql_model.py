import os
import sys
sys.path.extend([os.path.join(os.path.dirname(os.getcwd()), 'QA_wikisql')])
import json
import numpy as np

from stanford_parsing import replace_origin_preprocessing
import torch
from torch.utils.data import DataLoader

def make_origin_data(path):
    # read type question, header file
    types = ["train", "dev", "test"]
    for type in types:
        question_path = os.path.join(os.path.dirname(os.getcwd()), "WikiSQL/data/" + type + ".jsonl")
        table_path = os.path.join(os.path.dirname(os.getcwd()), "WikiSQL/data/" + type + ".tables.jsonl")

        qs = open(question_path).readlines()  # question sentence
        ts = open(table_path).readlines()  # table sentence

        # original data
        with open(os.path.join(path, type + ".jsonl"), "a", encoding="utf-8") as a:
            form = {"nli":"", "col":[], "table_id":"", "id":"", "name":""}
            for line in qs:
                doc = json.loads(line)
                form["nli"] += replace_origin_preprocessing(doc["question"])
                form["table_id"] += doc["table_id"]
                for table in ts:
                    table_doc = json.loads(table)
                    if table_doc['id'] == doc['table_id']:
                        try:
                            form["col"] = table_doc['header']
                            form["id"] += table_doc["id"]
                            form["name"] += table_doc["name"]
                        except KeyError as e:
                            break
                        break
                a.write(json.dumps(form, ensure_ascii=False))
                a.write("\n")
                form = {"nli":"", "col":[], "table_id":"", "id":"", "name":""}

        print("get {} original done.".format(type))

class preprocessed_dataset:
    raw_folder = "raw_data"

    # read image, label file
    training_set = (
        torch.from_numpy(np.array([1,2,3])).type(torch.FloatTensor),
        torch.from_numpy(np.array([1,2,3])).long()
    )
    test_set = (
        torch.from_numpy(np.array([10,20,30])).type(torch.FloatTensor),
        torch.from_numpy(np.array([10,20,30])).long()
    )

    def __init__(self, train=True):
        self.data = None
        self.labels = None

        if train == True:
            self.data, self.labels = self.training_set
        else:
            self.data, self.labels = self.test_set

    # __getitem__, __len__은 python의 규약(protocol) 일종(duck typing : if it walks like a duck and it quacks like a duck, then it must be a duck.)
    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        return img, target

    def __len__(self):
        return len(self.data)

class nl2sql_model():
    NotImplementedError

if __name__ == "__main__":
    path = os.path.join(os.path.dirname(os.getcwd()), 'QA_wikisql/preprocess')
    make_origin_data(path)
    with open(os.path.join(path, type + ".jsonl"), "r", encoding="utf-8") as r:
        data = r.readlines()
        for line in data:
            print(json.loads(line))
            break

    ## load data
    train_loader = torch.utils.data.DataLoader(preprocessed_dataset(type="train"), batch_size=1, shuffle=False)
    dev_loader = torch.utils.data.DataLoader(preprocessed_dataset(type="dev"), batch_size=1, shuffle=False)
    test_loader = torch.utils.data.DataLoader(preprocessed_dataset(type="test"), batch_size=1, shuffle=False)

    train_acc = 0
    for nl, col, sql in train_loader:
        print()

    test_acc = 0
    for nl, col, sql in test_loader:
        print()



