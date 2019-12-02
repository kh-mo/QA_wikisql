# QA with WikiSQL

## Task 설명
- 자연어를 통해 인간과 컴퓨터의 상호작용을 연구하는 분야, Natural Language Interface(NLI)
- 자연언어를 SQL 쿼리문으로 변화시켜주는 연구 분야(NL2SQL)

## 데이터 셋
- WikiSQL 데이터 셋 활용([링크](https://github.com/salesforce/WikiSQL))
- 80,654개 자연어 질문, 24,241개 테이블

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
### Do EDA
```shell
python wikisqlEDA.py
```

## Reference
- [1] [SEQ2SQL: GENERATING STRUCTURED QUERIES FROM NATURAL LANGUAGE USING REINFORCEMENT LEARNING](https://arxiv.org/pdf/1709.00103.pdf), arXiv 2017