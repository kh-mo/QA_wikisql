import os
import json

if __name__ == "__main__":
    train_path = os.path.join(os.path.dirname(os.getcwd()), 'WikiSQL/data/train.jsonl')
    dev_path = os.path.join(os.path.dirname(os.getcwd()), 'WikiSQL/data/dev.jsonl')
    test_path = os.path.join(os.path.dirname(os.getcwd()), 'WikiSQL/data/test.jsonl')

    with open(train_path) as tr, open(dev_path) as de, open(test_path) as ts:
        train_data = tr.readlines()
        dev_data = de.readlines()
        test_data = ts.readlines()

    print("train lines : {}, dev lines : {}, test lines : {}".format(len(train_data), len(dev_data), len(test_data)))
    print("dataset total : {}".format(len(train_data+dev_data+test_data)))
    # json.loads(train_data[0])

    train_table_path = os.path.join(os.path.dirname(os.getcwd()), 'WikiSQL/data/train.tables.jsonl')
    dev_table_path = os.path.join(os.path.dirname(os.getcwd()), 'WikiSQL/data/dev.tables.jsonl')
    test_table_path = os.path.join(os.path.dirname(os.getcwd()), 'WikiSQL/data/test.tables.jsonl')

    with open(train_table_path) as tr, open(dev_table_path) as de, open(test_table_path) as ts:
        train_table_data = tr.readlines()
        dev_table_data = de.readlines()
        test_table_data = ts.readlines()

    # 어떤 정보가 table정보를 담고 있는 것인지??
    tr_table = []
    dev_table = []
    ts_table = []
    criterion = "id"

    for doc in train_table_data:
        tr_table.append(json.loads(doc)[criterion].rsplit("-",maxsplit=1)[0])
    for doc in dev_table_data:
        dev_table.append(json.loads(doc)[criterion].rsplit("-",maxsplit=1)[0])
    for doc in test_table_data:
        ts_table.append(json.loads(doc)[criterion].rsplit("-",maxsplit=1)[0])

    tr_table = set(tr_table)
    dev_table = set(dev_table)
    ts_table = set(ts_table)

    print("train table lines : {}, dev table lines : {}, test table lines : {}".format(len(tr_table), len(dev_table), len(ts_table)))
    print("tables total : {}".format(len(tr_table|dev_table|ts_table)))
    len(set(tr_table) & set(dev_table))
    len(set(tr_table) & set(ts_table))
    len(set(dev_table) & set(ts_table))

    len(tr_table | dev_table | ts_table - (tr_table&dev_table) - (dev_table&ts_table) - (ts_table&tr_table)|(dev_table&ts_table&tr_table))

    len(tr_table & dev_table)
    len(dev_table & ts_table)
    len(ts_table&tr_table)
    len(tr_table & dev_table & ts_table)

    question_lengths = []
    query_lengths = []
    number_of_columns = []

    for line in (train_data+dev_data+test_data):
        question_lengths.append(len(json.loads(line)['question'].split(' ')))

    for line in (tr_table+dev_table+ts_table):
        question_lengths.append(len(json.loads(line)['question']))

    len(train_table_data+dev_table_data+test_table_data)
    import matplotlib.pyplot as plt
    plt.hist(question_lengths, bins=list(range(60)), range=(0,60))

    '''
    전체 학습 데이터 수 : 56,355
    전체 검증 데이터 수 : 8,421
    전체 테스트 데이터 수 : 15,878 
    전체 : 80,654
    
    학습 테이블 수 : 17,984
    검증 테이블 수 : 2,630
    테스트 테이블 수 : 5,069
    전체 : 25,683(페이퍼랑 다르다 - 더 많다?? 왜??)
    겹치는 테이블은 없다... 근데 어떻게 안 본 테이블에서 정보를 얻지?
    테이블 정보와 질문이 같이 들어간다는 가정
    '''

    json.loads(train_data[0])["table_id"].rsplit("-",maxsplit=1)[0]