# QA with WikiSQL

## Task 설명
- 자연어를 통해 인간과 컴퓨터의 상호작용을 연구하는 분야, Natural Language Interface(NLI)
- 자연언어를 SQL 쿼리문으로 변화시켜주는 연구 분야(NL2SQL)

## 데이터 셋
- WikiSQL 데이터 셋 활용([링크](https://github.com/salesforce/WikiSQL))
- 80,654개 자연어 질문, 24,241개 테이블
- Train, Dev 중복 table : 185개
- Dev, Test 중복 table : 57개
- Test, Train 중복 table : 426개
- Train, Dev, Test 동시에 중복되는 table은 없다

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

### Requirement
- python 3
- [WikiSQL](https://github.com/salesforce/WikiSQL)
- [corenlp for python](https://github.com/stanfordnlp/python-stanford-corenlp)

### Folder Structure
```
nli
 |--- QA_wikisql
 |--- WikiSQL
```

### Download Dataset
salesforce의 WikiSQL 깃 레포지토리로부터 데이터셋과 평가를 위한 코드를 다운로드 받는다.
```shell
from https://github.com/salesforce/WikiSQL

git clone https://github.com/salesforce/WikiSQL
cd WikiSQL
pip install -r requirements.txt
tar xvjf data.tar.bz2
```

### Do EDA
데이터 수, 테이블 수, [\[1\]](#Reference)에 작성된 figure 5의 question lengths, number of columns 그림 확인.
```shell
python wikisqlEDA.py
```

### Tokenizing
Dataset의 question과 table column name을 유형별(train, dev, test)로 모아 stanford parser로 tokenizing 진행.
이후 [\[2\]](#Reference), [\[3\]](#Reference) 알고리즘을 적용하여 강건한 input 제작.
```shell
python stanford_parsing.py
python bpe.py

this is example -> this __is __example -> th@@ is __is __ex@@ ample
```

### Restore
BPE와 stanford parser결과를 원래 문장으로 복원
공백제거 -> @@를 빈 공간으로 치환 -> __를 띄어쓰기로 치환
```shell
python restore.py

th@@ is __is __ex@@ ample -> th@@is__is__ex@@ample -> this__is__example
```

### Get Result
```shell
python evaluate.py --source_file=data/dev.jsonl --db_file=data/dev.db --pred_file=data/example.pred.dev.jsonl
```

## Reference
- [1] [SEQ2SQL: GENERATING STRUCTURED QUERIES FROM NATURAL LANGUAGE USING REINFORCEMENT LEARNING](https://arxiv.org/pdf/1709.00103.pdf), arXiv 2017
