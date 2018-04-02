##############################
## extractive summarization ##
##############################
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from gensim.models import Word2Vec

## load data
gensim_model = Word2Vec.load("C:/Users/user/Desktop/word2vec_model")
with open("C:/Users/user/Desktop/input_x", 'rb') as file:
    input_x = np.array(pickle.load(file))
with open("C:/Users/user/Desktop/input_y", 'rb') as file:
    input_y = np.array(pickle.load(file))

def make_batch(x, y, batch, word_dimension, gensim_model):
    batch_idx = np.random.choice(x.shape[0], batch, False)
    xs = x[batch_idx][:]
    ys = y[batch_idx][:]

    # 문서 하나에 단어를 쭉 나열해서 입력으로 사용
    # max_word_len 을 먼저 알아야 함
    max_word_len = 0
    max_seq_len = 0
    dyseq_length = []
    for content in xs:
        word_length_sum = 0
        tmp_seq_len = len(content)
        dyseq_length.append(len(content))
        for element in content:
            word_length_sum += len(element)
        if max_word_len < word_length_sum:
            max_word_len = word_length_sum
        if max_seq_len < tmp_seq_len:
            max_seq_len = tmp_seq_len

    # 단어 길이가 max보다 적은 문서는 0 패딩을 붙인다
    seq_eles = []
    seq_lens = []
    doc_idx = 0
    for content in xs:
        tmp_eles = []
        tmp_lens = []
        word_length_sum = 0
        for element in content:
            tmp_element = []
            for word in element:
                tmp_element.append(gensim_model[word].tolist())
            tmp_eles += tmp_element
            word_length_sum += len(element)
            tmp_lens += [[doc_idx, word_length_sum-1]]
        tmp_eles += ([[0.0]*word_dimension] * (max_word_len - len(tmp_eles)))
        for i in range(-(max_seq_len - len(tmp_lens)),0):
            tmp_lens += ([[doc_idx, max_word_len + i]])
        seq_eles.append(tmp_eles)
        seq_lens.append(tmp_lens)
        doc_idx += 1

    y_label = []
    for idx, label in enumerate(ys):
        tmp_ys = [[0,0]]*max_seq_len
        tmp_ys[:len(ys[idx])] = ys[idx]
        y_label.append(tmp_ys)

    return {model_x:seq_eles, dyseq_len:dyseq_length, model_y:y_label, sequence_length:seq_lens}

doc_batch = 50
number_of_document = None
number_of_sequence = None
number_of_word = None
word_dimension = 3
hidden_size = 3
output_dim = 2

model_x = tf.placeholder(dtype=tf.float32, shape=[number_of_document, number_of_word, word_dimension])
model_y = tf.placeholder(dtype=tf.float32, shape=[number_of_document, number_of_sequence, output_dim])
sequence_length = tf.placeholder(dtype=tf.int32, shape=[number_of_document, number_of_sequence, 2])
dyseq_len = tf.placeholder(dtype=tf.int32, shape=[None])

rnn_cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_size, activation=tf.tanh,
                                  kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                  bias_initializer=tf.contrib.layers.variance_scaling_initializer())
output, hid_output = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=model_x, sequence_length=dyseq_len, dtype=tf.float32)
sen_output = tf.gather_nd(output, sequence_length)

output_weight = tf.get_variable(name="output_weight", shape=[hidden_size, output_dim], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
output_bias = tf.get_variable(name="output_bias", shape=[output_dim], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
pred = tf.add(tf.tensordot(sen_output, output_weight, axes=[[2],[0]]), output_bias)
loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=model_y,logits=pred))
train = tf.train.AdamOptimizer(1e-4).minimize(loss)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for i in range(200000):
    batch = make_batch(input_x, input_y, doc_batch, word_dimension, gensim_model)
    pd, ls, tr = sess.run([pred, loss, train], feed_dict=batch)
    if i % 50 == 0:
        print("iteration :", i, "loss : ", ls)
    if i % 500 == 0:
        print(pd)


with open("C:/Users/user/Desktop/test_input_x", 'rb') as file:
    test_input_x = np.array(pickle.load(file))
with open("C:/Users/user/Desktop/test_input_y", 'rb') as file:
    test_input_y = np.array(pickle.load(file))


test_batch = test_make_batch(test_input_x, test_input_y, test_input_x.shape[0], word_dimension, gensim_model)
result = sess.run(tf.nn.softmax(pred), feed_dict=test_batch)

x = test_input_x
y = test_input_y
batch = test_input_x.shape[0]
def test_make_batch(x, y, batch, word_dimension, gensim_model):
    batch_idx = np.random.choice(x.shape[0], batch, False)
    xs = x[batch_idx][:]
    ys = y[batch_idx][:]

    # 문서 하나에 단어를 쭉 나열해서 입력으로 사용
    # max_word_len 을 먼저 알아야 함
    max_word_len = 0
    max_seq_len = 0
    dyseq_length = []
    for content in xs:
        word_length_sum = 0
        tmp_seq_len = len(content)
        dyseq_length.append(len(content))
        for element in content:
            word_length_sum += len(element)
        if max_word_len < word_length_sum:
            max_word_len = word_length_sum
        if max_seq_len < tmp_seq_len:
            max_seq_len = tmp_seq_len

    # 단어 길이가 max보다 적은 문서는 0 패딩을 붙인다
    seq_eles = []
    seq_lens = []
    doc_idx = 0
    for content in xs:
        tmp_eles = []
        tmp_lens = []
        word_length_sum = 0
        for element in content:
            tmp_element = []
            for word in element:
                try:
                    tmp_element.append(gensim_model[word].tolist())
                except (KeyError):
                    tmp_element.append([0,0,0])
            tmp_eles += tmp_element
            word_length_sum += len(element)
            tmp_lens += [[doc_idx, word_length_sum-1]]
        tmp_eles += ([[0.0]*word_dimension] * (max_word_len - len(tmp_eles)))
        for i in range(-(max_seq_len - len(tmp_lens)),0):
            tmp_lens += ([[doc_idx, max_word_len + i]])
        seq_eles.append(tmp_eles)
        seq_lens.append(tmp_lens)
        doc_idx += 1

    y_label = []
    for idx, label in enumerate(ys):
        tmp_ys = [[0,0]]*max_seq_len
        tmp_ys[:len(ys[idx])] = ys[idx]
        y_label.append(tmp_ys)

    return {model_x:seq_eles, dyseq_len:dyseq_length, model_y:y_label, sequence_length:seq_lens}


test_result = pd.DataFrame({"test_data": [], "pred_pos": [], "pred_neg": [], "real_pos": [], "real_neg": [], "doc_idx":[]})

for row_idx, row in enumerate(test_input_x):
    for sentence_idx, sentence in enumerate(row):
        d = {"test_data": [test_input_x[row_idx][sentence_idx]],
             "pred_pos": [result[row_idx][sentence_idx][0]],
             "pred_neg": [result[row_idx][sentence_idx][1]],
             "real_pos": [test_input_y[row_idx][sentence_idx][0]],
             "real_neg": [test_input_y[row_idx][sentence_idx][1]],
             "doc_idx": [row_idx]}
        test_result = test_result.append(pd.DataFrame(data=d))

test_result.to_csv("C:/Users/user/Desktop/Final.csv")
tf.train.Saver().save(sess, "C:/Users/user/Desktop/extractive_model")