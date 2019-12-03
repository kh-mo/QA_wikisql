import os
import json
import matplotlib.pyplot as plt

if __name__ == "__main__":
    ## question information
    train_path = os.path.join(os.path.dirname(os.getcwd()), 'WikiSQL/data/train.jsonl')
    dev_path = os.path.join(os.path.dirname(os.getcwd()), 'WikiSQL/data/dev.jsonl')
    test_path = os.path.join(os.path.dirname(os.getcwd()), 'WikiSQL/data/test.jsonl')

    with open(train_path) as tr, open(dev_path) as de, open(test_path) as ts:
        train_data = tr.readlines()
        dev_data = de.readlines()
        test_data = ts.readlines()

    print("train lines : {}, dev lines : {}, test lines : {}".format(len(train_data), len(dev_data), len(test_data)))
    print("dataset total : {}".format(len(train_data+dev_data+test_data)))

    ## table information
    tr_table = []
    dev_table = []
    ts_table = []
    criterion = "table_id"

    for doc in train_data:
        tr_table.append(json.loads(doc)[criterion].split("-",maxsplit=1)[1])
    for doc in dev_data:
        dev_table.append(json.loads(doc)[criterion].split("-",maxsplit=1)[1])
    for doc in test_data:
        ts_table.append(json.loads(doc)[criterion].split("-",maxsplit=1)[1])

    tr_table = set(tr_table)
    dev_table = set(dev_table)
    ts_table = set(ts_table)

    print("train table lines : {}, dev table lines : {}, test table lines : {}".format(len(tr_table), len(dev_table), len(ts_table)))
    print("tables total : {}".format(len(tr_table|dev_table|ts_table)))

    ## plot
    question_lengths = []
    query_lengths = []
    number_of_columns = []

    for line in (train_data+dev_data+test_data):
        question_lengths.append(len(json.loads(line)['question'].split(' ')))

    train_table_path = os.path.join(os.path.dirname(os.getcwd()), 'WikiSQL/data/train.tables.jsonl')
    dev_table_path = os.path.join(os.path.dirname(os.getcwd()), 'WikiSQL/data/dev.tables.jsonl')
    test_table_path = os.path.join(os.path.dirname(os.getcwd()), 'WikiSQL/data/test.tables.jsonl')

    with open(train_table_path) as tr, open(dev_table_path) as de, open(test_table_path) as ts:
        train_table_data = tr.readlines()
        dev_table_data = de.readlines()
        test_table_data = ts.readlines()

    for line in (train_table_data + dev_table_data + test_table_data):
        file = json.loads(line)
        if file["id"].split("-",maxsplit=1)[1] in list(tr_table | dev_table | ts_table):
            number_of_columns.append(len(file["header"]))

    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1) # row, col, index
    ax2 = fig.add_subplot(1, 2, 2)

    ax1.hist(question_lengths, bins=list(range(60)), range=(0,60))
    ax1.set_xlabel("Question lengths")
    ax1.set_ylabel("Frequency")

    ax2.hist(number_of_columns, bins=list(range(40)), range=(0, 40))
    ax2.set_xlabel("Number of columns")
    ax2.set_ylabel("Frequency")