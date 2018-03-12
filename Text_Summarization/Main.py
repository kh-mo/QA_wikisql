##############################
## extractive summarization ##
##############################
import numpy as np
import tensorflow as tf

def make_batch(data, doc_batch):
    batch_idx = np.random.choice(data.shape[0], doc_batch, False)
    xs = data[batch_idx]
    ys = label[batch_idx]

    max_word_len = 0
    for content in xs:
        word_length_sum = 0
        for element in content:
            word_length_sum += len(element)
        if max_word_len < word_length_sum:
            max_word_len = word_length_sum

    seq_eles = []
    seq_lens = []
    doc_idx = 0
    for content in xs:
        tmp_eles = []
        tmp_lens = []
        word_length_sum = 0
        for element in content:
            tmp_eles += element
            word_length_sum += len(element)
            tmp_lens += [[doc_idx,word_length_sum-1]]
        tmp_eles += ([[0]] * (max_word_len - len(tmp_eles)))
        seq_eles.append(tmp_eles)
        seq_lens.append(tmp_lens)
        doc_idx += 1
    return {doc_x: seq_eles, sequence_length: seq_lens, y:ys}

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

doc_batch = 2
number_of_document = None
number_of_sequence = None
number_of_word = None
word_dimension = 1
hidden_size = 4
output_dim = 2

doc_x = tf.placeholder(dtype=tf.float32, shape=[number_of_document, number_of_word, word_dimension])
y = tf.placeholder(dtype=tf.float32, shape=[number_of_document, number_of_sequence, output_dim])
sequence_length = tf.placeholder(dtype=tf.int32, shape=[number_of_document, number_of_sequence, 2])

rnn_cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_size, activation=tf.tanh,
                                  kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                  bias_initializer=tf.contrib.layers.variance_scaling_initializer())
output, hid_output = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=doc_x, dtype=tf.float32)
sen_output = tf.gather_nd(output, sequence_length)

output_weight = tf.get_variable(name="output_weight", shape=[hidden_size, output_dim], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
output_bias = tf.get_variable(name="output_bias", shape=[output_dim], dtype=tf.float32, initializer=tf.contrib.layers.variance_scaling_initializer())
pred = tf.add(tf.tensordot(sen_output, output_weight, axes=[[2],[0]]), output_bias)
loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=pred))
train = tf.train.AdamOptimizer(1e-4).minimize(loss)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
# a = make_batch(sequence, doc_batch)
# sess.run(sen_output, feed_dict=a).shape
for i in range(100000):
    pd, ls, tr = sess.run([pred, loss, train], feed_dict=make_batch(sequence, doc_batch))
    if i % 50 == 0:
        print(i, pd, ls)

