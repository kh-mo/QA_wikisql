import json
import argparse
from tqdm import tqdm
from lib.query import Query
from lib.dbengine import DBEngine
from lib.common import count_lines

# import os
# a = os.path.join(os.path.dirname(os.getcwd()), 'WikiSQL/data/dev.jsonl')
# b = os.path.join(os.path.dirname(os.getcwd()), 'WikiSQL/test/example.pred.dev.jsonl')
# c = os.path.join(os.path.dirname(os.getcwd()), 'WikiSQL/data/dev.db')
# engine = DBEngine(c)
# with open(a) as sf, open(b) as pf:
#     for source_line, pred_line in tqdm(zip(sf, pf), total=count_lines(a)):
#         # line별 정답과 예측 샘플 가져오기
#         gold_example = json.loads(source_line)
#         pred_example = json.loads(pred_line)
#         lf_gold_query = Query.from_dict(gold_example['sql'], ordered=True)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_file")
    parser.add_argument("--db_file")
    parser.add_argument("--pred_file")
    parser.add_argument("--ordered")
    args = parser.parse_args()

    engine = DBEngine(args.db_file)
    ex_acc_list = []
    lf_acc_list = []
    with open(args.source_file) as sf, open(args.pred_file) as pf:
        for source_line, pred_line in tqdm(zip(sf, pf), total=count_lines(args.source_file)):
            # line별 정답과 예측 샘플 가져오기
            gold_example = json.loads(source_line)
            pred_example = json.loads(pred_line)

            # 정답 샘플 lf, ex 구하기
            lf_gold_query = Query.from_dict(gold_example['sql'], ordered=args.ordered)
            ex_gold = engine.execute_query(gold_example['table_id'], lf_gold_query, lower=True)

            # error가 아닌 경우 예측 샘플 lf, ex 구하기
            lf_pred_query = None
            ex_pred = pred_example.get('error', None)
            if not ex_pred:
                try:
                    lf_pred_query = Query.from_dict(pred_example['query'], ordered=args.ordered)
                    ex_pred = engine.execute_query(gold_example['table_id'], lf_pred_query, lower=True)
                except Exception as e:
                    ex_pred = repr(e)

            # lf, ex의 gold, pred 매칭결과 구하기
            ex_acc_list.append(ex_pred == ex_gold)
            lf_acc_list.append(lf_pred_query == lf_gold_query)
            print('ex_accuracy {}\n lf_accuracy {}'.format(
                sum(ex_acc_list)/len(ex_acc_list), sum(lf_acc_list)/len(lf_acc_list),))
