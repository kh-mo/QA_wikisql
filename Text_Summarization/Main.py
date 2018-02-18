import numpy as np
import tensorflow as tf

batch_size = 64
word_dimension = 100
voca_size = 150000
sen_num = 100
sen_length = 50
hidden_size = 200

x = tf.placeholder(dtype=tf.float32, shape=[batch_size, sen_length, word_dimension], name="input_x")
y = tf.placeholder(dtype=tf.float32, shape=[], name="label")

word_level_forward_gru = tf.nn.rnn_cell.GRUCell(num_units=hidden_size,
                                                activation=tf.tanh,
                                                kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                                bias_initializer=tf.contrib.layers.variance_scaling_initializer())
word_level_forward_hidden_state = tf.zeros([batch_size, hidden_size])
word_level_backward_gru = tf.nn.rnn_cell.GRUCell(num_units=hidden_size,
                                                 activation=tf.tanh,
                                                 kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                                 bias_initializer=tf.contrib.layers.variance_scaling_initializer())
word_level_backward_hidden_state = tf.zeros([batch_size, hidden_size])
output, hidden_output = tf.nn.bidirectional_dynamic_rnn(cell_fw=word_level_forward_gru,
                                                        cell_bw=word_level_backward_gru,
                                                        inputs=x,
                                                        initial_state_fw=word_level_forward_hidden_state,
                                                        initial_state_bw=word_level_backward_hidden_state,
                                                        dtype=tf.float32)
sen_input = tf.tanh(tf.matmul(tf.reduce_mean(tf.concat(output,2),axis=1),
                              tf.truncated_normal([400, hidden_size], mean=0, stddev=1))
                    + tf.Variable(tf.constant(0.1, shape=[hidden_size])))

sentence_level_forward_gru = tf.nn.rnn_cell.GRUCell(num_units=hidden_size,
                                                    activation=tf.tanh,
                                                    kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                                    bias_initializer=tf.contrib.layers.variance_scaling_initializer())
sentence_level_forward_hidden_state = tf.zeros([batch_size, hidden_size])
sentence_level_backward_gru = tf.nn.rnn_cell.GRUCell(num_units=hidden_size,
                                                     activation=tf.tanh,
                                                     kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
                                                     bias_initializer=tf.contrib.layers.variance_scaling_initializer())
sentence_level_backward_hidden_state = tf.zeros([batch_size, hidden_size])
sen_output, sen_hidden_output = tf.nn.bidirectional_dynamic_rnn(cell_fw=sentence_level_forward_gru,
                                                        cell_bw=sentence_level_backward_gru,
                                                        inputs=sen_input,
                                                        initial_state_fw=sentence_level_forward_hidden_state,
                                                        initial_state_bw=sentence_level_backward_hidden_state,
                                                        dtype=tf.float32)


