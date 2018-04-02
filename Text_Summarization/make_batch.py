import pickle
import datetime
import pandas as pd
from itertools import chain
from gensim.models import Word2Vec
pd.options.mode.chained_assignment = None

## base_time부터 dates만큼 데이터를 읽어와서 cummulated_data에 저장
dates = range(1)
base_time = datetime.datetime(2018, 3, 28)
cummulated_data = pd.DataFrame({"title": [], "date": [], "body": [], "sum_content": [], "company": []})

for date in dates:
    if date%100==0:
        print(date)
    dif_time = datetime.timedelta(days=date)
    present_time = (base_time - dif_time).strftime("%Y%m%d")
    current_data = pd.read_csv("C:/Users/user/Desktop/crawling/crawled_book_"+present_time+".csv", encoding='CP949')
    ## 의미없는 unnamed: 0 열 삭제
    current_data = current_data.drop('Unnamed: 0', axis=1)
    ## 데이터가 없는 NA 행 삭제
    current_data = current_data.dropna(axis=0)

    ## 본문의 첫 단어와 요약문의 첫 단어가 일치하지 않으면 잘못 수집한 데이터이브로 해당 index 수집하여 삭제
    drop_idx =[]
    for i in range(current_data.shape[0]):
        body_first_word = current_data['body'].iloc[i].split()[0]
        summary_first_word = current_data['sum_content'].iloc[i].split()[0]
        current_data['body'].iloc[i] = current_data['body'].iloc[i].replace('.', '. <EOS><cut> ')
        current_data['sum_content'].iloc[i] = current_data['sum_content'].iloc[i].replace('.', '. <EOS><cut> ')
        if (len(current_data['body'].iloc[i].split('<cut>')) <= 3) | (body_first_word != summary_first_word):
            drop_idx.append(i)

    current_data = current_data.drop(current_data.index[drop_idx])
    cummulated_data = cummulated_data.append(current_data)

cummulated_data.shape
cummulated_data.to_csv("C:/Users/user/Desktop/crawling/all/test.csv", index=False)

## 잘 수집된 cummulated_data로부터 input_x와 input_y를 만드는 과정
x = []
y = []
word_list = []
word2vec_list = []

for row_idx in range(cummulated_data.shape[0]):
    body = cummulated_data['body'].iloc[row_idx].split('<cut>')[:-1]
    summary = cummulated_data['sum_content'].iloc[row_idx].split('<cut>')[:-1]

    doc_x = []
    doc_y = []

    for element in body:
        tmp = [1,0] if element in summary else [0,1]
        splited_element = element.split()
        doc_x.append(splited_element)
        doc_y.append(tmp)
        word_list.append(splited_element)
        word2vec_list.append(splited_element)

    x.append(doc_x)
    y.append(doc_y)

    if row_idx % 100 == 0:
        print(row_idx)
        word_list = list(set(chain(*word_list)))

gensim_model = Word2Vec(word2vec_list, size=3, window=5, min_count=1, sg=1)
gensim_model["<EOS>"]

## save file
# gensim_model save
gensim_model.save("C:/Users/user/Desktop/word2vec_model")
# save input_x
with open("C:/Users/user/Desktop/test_input_x", 'wb') as file:
    pickle.dump(x, file)
# save input_y
with open("C:/Users/user/Desktop/test_input_y", 'wb') as file:
    pickle.dump(y, file)


'''
sequence = np.asarray([[[[1]],
                        [[2],[3]],
                        [[4],[5],[6]]],
                       [[[11],[12],[13],[14]],
                        [[14],[15]],
                        [[16]]]])

label = np.asarray([[[1,0],
                     [0,1],
                     [0,1]],
                    [[1,0],
                     [0,1],
                     [0,1]]])
'''