# QA with WikiSQL

## Task 설명
- 자연어를 통해 인간과 컴퓨터의 상호작용을 연구하는 분야, Natural Language Interface(NLI)
- 자연언어를 SQL 쿼리문으로 변화시켜주는 연구 분야(NL2SQL)

## 데이터 셋
- WikiSQL 데이터 셋 활용([링크](https://github.com/salesforce/WikiSQL))
- 80,654개 자연어 질문, 24,241개 테이블
- Train, Dev, Test table 사이에 중복된 테이블은 없다
- 페이퍼보다 table이 좀 더 많다?? 그 이유는?? (2290개 더 많다)

*데이터 셋* | *Questions* | *SQL tables* |
:---: | :---: | :---: |
Train | 56,355개 | 17,290개 |
Dev | 8,421개 | 2,615개 |
Test | 15,878개 | 5,004개 |
Total | 80,654개 | 24,241개 |

## 평가
### 리더보드
*모델* | *Dev LFA* | *Dev EA* | *Test LFA* | *Test EA* |
:---: | :---: | :---: | :---: | :---: |
model1 | 0.0 | 0.0 | 0.0 | 0.0 |
baseline | 0.0 | 0.0 | 0.0 | 0.0 |

### 평가지표
- Execution Accuracy : 쿼리 실행 결과가 정확한 결과를 반환하는지 여부
- Logical Form Accuary : 쿼리문이 정답과 일치하는 여부

## Getting Start

### requirement
- python 3
- [WikiSQL](https://github.com/salesforce/WikiSQL)
- [corenlp for python](https://github.com/stanfordnlp/python-stanford-corenlp)

### Download Dataset
```
from https://github.com/salesforce/WikiSQL

git clone https://github.com/salesforce/WikiSQL
cd WikiSQL
pip install -r requirements.txt
tar xvjf data.tar.bz2
```

### Do EDA
```shell
python wikisqlEDA.py
```

### Tokenizing
```shell
python stanford_parsing.py
python bpe.py
```

### Get Result
```shell
python evaluate.py --source_file=data/dev.jsonl --db_file=data/dev.db --pred_file=data/example.pred.dev.jsonl
```

## Reference
- [1] [SEQ2SQL: GENERATING STRUCTURED QUERIES FROM NATURAL LANGUAGE USING REINFORCEMENT LEARNING](https://arxiv.org/pdf/1709.00103.pdf), arXiv 2017
