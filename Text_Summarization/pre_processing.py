import pandas as pd

raw_data = pd.read_csv("C:/Users/Mo/Desktop/행정.csv", engine='python')

raw_data = raw_data[raw_data['body']!=" "]
raw_data = raw_data.dropna(axis=0, how='any')
raw_data = raw_data[raw_data['sum_content']!="해당 기사는 요청하신 자동 요약문을 제공할 수 없습니다. "]

len(raw_data)

result = pd.DataFrame({"doc" : [], "sen_idx" : [], "sentence" : []})
for doc in range(len(raw_data)):
    print(doc)
    try:
        sentence_list = raw_data['body'][doc].split('\n')
    except KeyError:
        pass
    for sen_idx, sentence in enumerate(sentence_list):
        d = {"doc" : [doc], "sen_idx" : [sen_idx], "sentence" : [sentence]}
        result = result.append(pd.DataFrame(data=d))

result = result[result['sentence']!=" "]
result = result[result['sentence']!=""]
result = result[result['sentence']!="\t"]
result = result[result['sentence']!="??"]
result = result[result['sentence']!="[페이스북]"]
result = result[result['sentence']!="[트위터]"]
result = result[result['sentence']!="[사진 영상 제보받습니다]"]
result = result[result['sentence']!="[카카오 친구맺기]"]
result = result[result['sentence']!="[카카오 플러스친구]"]
result = result[result['sentence']!="[모바일웹]"]

result.to_csv("C:/Users/mo/Desktop/test.csv", index=False)


















len(result)
result = result[result['sentence']==re.compile("[A-Za-z0-9\._+]+@[A-Za-z]+\.(com|org|edu|net)")]

result['sentence']==regex
regex = re.compile("[A-Za-z0-9\._+]+@[A-Za-z]+\.(com|org|edu|net)", flags=re.IGNORECASE)
print(regex.match('wesm@bright.net'))



import re
re.compile
result[result['sentence']=="^▶+"]





raw_data['body'][0].split('\n')
raw_data['body'][1].split('\n')
raw_data['body'][2].split('\n')
raw_data['body'][3].split('\n')
raw_data['body'][4].split('\n')
raw_data['body'][5].split('\n')
raw_data['body'][6].split('\n')
raw_data['body'][7].split('\n')