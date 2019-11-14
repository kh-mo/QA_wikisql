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

    print(len(train_data))
    print(len(dev_data))
    print(len(test_data))

    tr_table = []
    dev_table = []
    ts_table = []

    for doc in train_data:
        tr_table.append(json.loads(doc)['table_id'])
    for doc in dev_data:
        dev_table.append(json.loads(doc)['table_id'])
    for doc in test_data:
        ts_table.append(json.loads(doc)['table_id'])

    tr_table = list(set(tr_table))
    dev_table = list(set(dev_table))
    ts_table = list(set(ts_table))

    print(len(tr_table))
    print(len(dev_table))
    print(len(ts_table))

    print(len(train_data) + len(dev_data) + len(test_data))
    print(len(tr_table) + len(dev_table) + len(ts_table))

    list(set(tr_table) & set(dev_table))
    list(set(tr_table) & set(ts_table))
    list(set(dev_table) & set(ts_table))
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