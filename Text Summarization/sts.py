import os
import sys
sys.path.append(os.getcwd()+"/Text Summarization/")
import time
import numpy as np
import tool as tool
import tensorflow as tf
import pandas as pd

# data loading
data_path = 'C:/Users/user/Desktop/2016newscorpus2.csv'

def loading_data(data_path):
    # R에서 title과 contents만 csv로 저장한걸 불러와서 제목과 컨텐츠로 분리
    # write.csv(corpus, data_path, fileEncoding='utf-8', row.names=F)
    corpus = np.array(pd.read_table(data_path, sep=",", encoding="utf-8"))
    title = []
    body = []
    for idx, doc in enumerate(corpus):
        title.append(doc[0].split())
        body.append(doc[1].split())
        if idx % 100000 == 0:
            print('%d docs' % (idx))
    return title, body

title, contents = loading_data(data_path)
input = title+contents

def convert_dic(input):
    uniq_word = []
    for i in range(len(input)):
        uniq_word = list(set(uniq_word+input[i]))
    uniq_word.sort()
    idx = list(range(len(uniq_word)))
    word_to_idx = dict(zip(uniq_word, idx))
    idx_to_word = dict(zip(idx, uniq_word))
    return word_to_idx, idx_to_word

# word_to_ix, ix_to_word = tool.make_dict_all_cut(title+contents, minlength=0, maxlength=3, jamo_delete=True)
word_to_idx, idx_to_word = convert_dic(title + contents)

# parameters
multi = True
forward_only = False ##training False / testing True
hidden_size = 300
vocab_size = len(idx_to_word)
num_layers = 3
learning_rate = 0.001
batch_size = 16
encoder_size = 10
# decoder_size = tool.check_doclength(title,sep=True) # (Maximum) number of time steps in this batch
decoder_size = 5
steps_per_checkpoint = 10

# transform data
encoderinputs, decoderinputs, targets_, targetweights = tool.make_inputs(contents, title, word_to_idx,
                     encoder_size=encoder_size, decoder_size=decoder_size, shuffle=False)

class seq2seq(object):

    def __init__(self, multi, hidden_size, num_layers, forward_only, learning_rate,
                 batch_size, vocab_size, encoder_size, decoder_size):

        # variables
        self.source_vocab_size = vocab_size
        self.target_vocab_size = vocab_size
        self.batch_size = batch_size
        self.encoder_size = encoder_size
        self.decoder_size = decoder_size
        self.learning_rate = tf.Variable(float(learning_rate), trainable=False, name="learning_rate")
        self.global_step = tf.Variable(0, trainable=False, name="global_step")

        source_vocab_size = vocab_size
        target_vocab_size = vocab_size
        batch_size = batch_size
        encoder_size = encoder_size
        decoder_size = decoder_size
        learning_rate = tf.Variable(float(learning_rate), trainable=False, name="learning_rate")
        global_step = tf.Variable(0, trainable=False, name="global_step")

        # networks
        W = tf.Variable(tf.random_normal([hidden_size, vocab_size]), name="output_to_hidden_weight")
        b = tf.Variable(tf.random_normal([vocab_size]), name="output_to_hidden_bias")
        output_projection = (W, b)
        self.encoder_inputs = [tf.placeholder(tf.int32, [batch_size], name="encoder_inputs"+str(_)) for _ in range(encoder_size)]  # 인덱스만 있는 데이터 (원핫 인코딩 미시행)
        self.decoder_inputs = [tf.placeholder(tf.int32, [batch_size], name="decoder_inputs"+str(_)) for _ in range(decoder_size)]
        self.targets = [tf.placeholder(tf.int32, [batch_size], name="targets"+str(_)) for _ in range(decoder_size)]
        self.target_weights = [tf.placeholder(tf.float32, [batch_size], name="target_weight"+str(_)) for _ in range(decoder_size)]

        encoder_inputs = [tf.placeholder(tf.float32, [batch_size, source_vocab_size], name="encoder_inputs" + str(_)) for _ in
                               range(encoder_size)]  # 인덱스만 있는 데이터 (원핫 인코딩 미시행)
        decoder_inputs = [tf.placeholder(tf.float32, [batch_size, target_vocab_size], name="decoder_inputs" + str(_)) for _ in
                               range(decoder_size)]
        targets = [tf.placeholder(tf.float32, [batch_size], name="targets" + str(_)) for _ in range(decoder_size)]
        target_weights = [tf.placeholder(tf.float32, [batch_size], name="target_weight" + str(_)) for _ in
                               range(decoder_size)]
        # models
        if multi:
            # single_cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_size)
            single_cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size)
            cell = tf.nn.rnn_cell.MultiRNNCell([single_cell, single_cell])
            # cell = tf.nn.rnn_cell.MultiRNNCell([single_cell]*num_layers, state_is_tuple=True)
        else:
            cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_size)
            #cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size)

        if not forward_only:
            self.outputs, self.states = tf.nn.seq2seq.embedding_attention_seq2seq(
                self.encoder_inputs, self.decoder_inputs, cell,
                num_encoder_symbols=vocab_size,
                num_decoder_symbols=vocab_size,
                embedding_size=hidden_size,
                output_projection=output_projection,
                feed_previous=False)


            outputs, states = tf.contrib.legacy_seq2seq.basic_rnn_seq2seq(encoder_inputs, decoder_inputs, single_cell)
            outputs, states = tf.contrib.legacy_seq2seq.basic_rnn_seq2seq(encoder_inputs, decoder_inputs, cell)


            logits = [tf.matmul(output_element, output_projection[0]) + output_projection[1] for output_element in outputs]

            outputs, states = tf.nn.seq2seq.embedding_attention_seq2seq(encoder_inputs, decoder_inputs, cell,
                                                                        num_encoder_symbols = vocab_size,
                                                                        num_decoder_symbols = vocab_size,
                                                                        embedding_size = hidden_size,
                                                                        output_projection = output_projection,
                                                                        feed_previous = False)


            self.logits = [tf.matmul(output, output_projection[0]) + output_projection[1] for output in self.outputs]
            self.loss = []
            for logit, target, target_weight in zip(self.logits, self.targets, self.target_weights):
                crossentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logit, target)
                self.loss.append(crossentropy * target_weight)
            self.cost = tf.nn.math_ops.add_n(self.loss)
            self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)


        else:
            self.outputs, self.states = tf.nn.seq2seq.embedding_attention_seq2seq(
                self.encoder_inputs, self.decoder_inputs, cell,
                num_encoder_symbols=vocab_size,
                num_decoder_symbols=vocab_size,
                embedding_size=hidden_size,
                output_projection=output_projection,
                feed_previous=True)
            self.logits = [tf.matmul(output, output_projection[0]) + output_projection[1] for output in self.outputs]

    def step(self, session, encoderinputs, decoderinputs, targets, targetweights, forward_only):
        input_feed = {}
        for l in range(len(encoder_inputs)):
            input_feed[self.encoder_inputs[l].name] = encoderinputs[l]
        for l in range(len(decoder_inputs)):
            input_feed[self.decoder_inputs[l].name] = decoderinputs[l]
            input_feed[self.targets[l].name] = targets[l]
            input_feed[self.target_weights[l].name] = targetweights[l]
        if not forward_only:
            output_feed = [self.train_op, self.cost]
        else:
            output_feed = []
            for l in range(len(decoder_inputs)):
                output_feed.append(self.logits[l])
        output = session.run(output_feed, input_feed)
        if not forward_only:
            return output[1] # loss
        else:
            return output[0:] # outputs

sess = tf.Session()
train_writer = tf.summary.FileWriter("C:/Users/user/Desktop/tmp", sess.graph)
model = seq2seq(multi=multi, hidden_size=hidden_size, num_layers=num_layers, learning_rate=learning_rate,
                batch_size=batch_size, vocab_size=vocab_size, encoder_size=encoder_size, decoder_size=decoder_size,
                forward_only=forward_only)
sess.run(tf.global_variables_initializer())
step_time, loss = 0.0, 0.0
current_step = 0
start = 0
end = batch_size
while current_step < 10000001:

    if end > len(title):
        start = 0
        end = batch_size

    # Get a batch and make a step
    start_time = time.time()
    encoder_inputs, decoder_inputs, targets, target_weights = tool.make_batch(encoderinputs[start:end],
                                                                              decoderinputs[start:end],
                                                                              targets_[start:end],
                                                                              targetweights[start:end])

    if current_step % steps_per_checkpoint == 0:
        for i in range(decoder_size - 2):
            decoder_inputs[i + 1] = np.array([word_to_ix['<PAD>']] * batch_size)
        output_logits = model.step(sess, encoder_inputs, decoder_inputs, targets, target_weights, True)
        predict = [np.argmax(logit, axis=1)[0] for logit in output_logits]
        predict = ' '.join(ix_to_word[ix][0] for ix in predict)
        real = [word[0] for word in targets]
        real = ' '.join(ix_to_word[ix][0] for ix in real)
        print('\n----\n step : %s \n time : %s \n LOSS : %s \n 예측 : %s \n 손질한 정답 : %s \n 정답 : %s \n----' %
              (current_step, step_time, loss, predict, real, title[start]))
        loss, step_time = 0.0, 0.0

    step_loss = model.step(sess, encoder_inputs, decoder_inputs, targets, target_weights, False)
    step_time += time.time() - start_time / steps_per_checkpoint
    loss += np.mean(step_loss) / steps_per_checkpoint
    current_step += 1
    start += batch_size
    end += batch_size

import numpy as np
import pandas as pd
from collections import defaultdict


####################################################
# loading function                                 #
####################################################

def loading_data(data_path, eng=True, num=True, punc=False):
    # data example : "title","content"
    # data format : csv, utf-8
    corpus = pd.read_table(data_path, sep=",", encoding="utf-8")
    corpus = np.array(corpus)
    title = []
    contents = []
    for doc in corpus:
        if type(doc[0]) is not str or type(doc[1]) is not str:
            continue
        if len(doc[0]) > 0 and len(doc[1]) > 0:
            tmptitle = normalize(doc[0], english=eng, number=num, punctuation=punc)
            tmpcontents = normalize(doc[1], english=eng, number=num, punctuation=punc)
            title.append(tmptitle)
            contents.append(tmpcontents)
    return title, contents


def make_dict_all_cut(contents, minlength, maxlength, jamo_delete=False):
    dict = defaultdict(lambda: [])
    for doc in contents:
        for idx, word in enumerate(doc.split()):
            if len(word) > minlength:
                normalizedword = word[:maxlength]
                if jamo_delete:
                    tmp = []
                    for char in normalizedword:
                        if ord(char) < 12593 or ord(char) > 12643:
                            tmp.append(char)
                    normalizedword = ''.join(char for char in tmp)
                if word not in dict[normalizedword]:
                    dict[normalizedword].append(word)
    dict = sorted(dict.items(), key=operator.itemgetter(0))[1:]
    words = []
    for i in range(len(dict)):
        word = []
        word.append(dict[i][0])
        for w in dict[i][1]:
            if w not in word:
                word.append(w)
        words.append(word)

    words.append(['<PAD>'])
    words.append(['<S>'])
    words.append(['<E>'])
    words.append(['<UNK>'])
    # word_to_ix, ix_to_word 생성
    ix_to_word = {i: ch[0] for i, ch in enumerate(words)}
    word_to_ix = {}
    for idx, words in enumerate(words):
        for word in words:
            word_to_ix[word] = idx
    print('컨텐츠 갯수 : %s, 단어 갯수 : %s'
          % (len(contents), len(ix_to_word)))
    return word_to_ix, ix_to_word


####################################################
# making input function                            #
####################################################

def make_inputs(rawinputs, rawtargets, word_to_ix, encoder_size, decoder_size, shuffle=True):
    rawinputs = np.array(rawinputs)
    rawtargets = np.array(rawtargets)
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(len(rawinputs)))
        rawinputs = rawinputs[shuffle_indices]
        rawtargets = rawtargets[shuffle_indices]
    encoder_input = []
    decoder_input = []
    targets = []
    target_weights = []
    for rawinput, rawtarget in zip(rawinputs, rawtargets):
        tmp_encoder_input = [word_to_ix[v] for idx, v in enumerate(rawinput.split()) if
                             idx < encoder_size and v in word_to_ix]
        encoder_padd_size = max(encoder_size - len(tmp_encoder_input), 0)
        encoder_padd = [word_to_ix['<PAD>']] * encoder_padd_size
        encoder_input.append(list(reversed(tmp_encoder_input + encoder_padd)))
        tmp_decoder_input = [word_to_ix[v] for idx, v in enumerate(rawtarget.split()) if
                             idx < decoder_size - 1 and v in word_to_ix]
        decoder_padd_size = decoder_size - len(tmp_decoder_input) - 1
        decoder_padd = [word_to_ix['<PAD>']] * decoder_padd_size
        decoder_input.append([word_to_ix['<S>']] + tmp_decoder_input + decoder_padd)
        targets.append(tmp_decoder_input + [word_to_ix['<E>']] + decoder_padd)
        tmp_targets_weight = np.ones(decoder_size, dtype=np.float32)
        tmp_targets_weight[-decoder_padd_size:] = 0
        target_weights.append(list(tmp_targets_weight))
    return encoder_input, decoder_input, targets, target_weights


####################################################
# doclength check function                         #
####################################################
def check_doclength(docs, sep=True):
    max_document_length = 0
    for doc in docs:
        if sep:
            words = doc.split()
            document_length = len(words)
        else:
            document_length = len(doc)
        if document_length > max_document_length:
            max_document_length = document_length
    return max_document_length


####################################################
# making batch function                            #
####################################################
def make_batch(encoder_inputs, decoder_inputs, targets, target_weights):
    encoder_size = len(encoder_inputs[0])
    decoder_size = len(decoder_inputs[0])
    encoder_inputs, decoder_inputs, targets, target_weights = \
        np.array(encoder_inputs), np.array(decoder_inputs), np.array(targets), np.array(target_weights)
    result_encoder_inputs = []
    result_decoder_inputs = []
    result_targets = []
    result_target_weights = []
    for i in range(encoder_size):
        result_encoder_inputs.append(encoder_inputs[:, i])
    for j in range(decoder_size):
        result_decoder_inputs.append(decoder_inputs[:, j])
        result_targets.append(targets[:, j])
        result_target_weights.append(target_weights[:, j])
    return result_encoder_inputs, result_decoder_inputs, result_targets, result_target_weights









####################################################################################
import numpy as np
import pandas as pd
import tensorflow as tf

# data loading
data_path = 'C:/Users/user/Desktop/2016newscorpus2.csv'

def loading_data(data_path):
    # R에서 title과 contents만 csv로 저장한걸 불러와서 제목과 컨텐츠로 분리
    # write.csv(corpus, data_path, fileEncoding='utf-8', row.names=F)
    corpus = np.array(pd.read_table(data_path, sep=",", encoding="utf-8"))
    title = []
    body = []
    for idx, doc in enumerate(corpus):
        title.append(doc[0].split())
        body.append(doc[1].split())
        if idx % 100000 == 0:
            print('%d docs' % (idx))
    return np.asarray(title), np.asarray(body)

title, contents = loading_data(data_path)
input = title + contents

def convert_dic(input):
    uniq_word = []
    for i in range(len(input)):
        uniq_word = list(set(uniq_word+input[i]))
    uniq_word.sort()
    idx = list(range(len(uniq_word)))
    word_to_idx = dict(zip(uniq_word, idx))
    idx_to_word = dict(zip(idx, uniq_word))
    return word_to_idx, idx_to_word

# word_to_ix, ix_to_word = tool.make_dict_all_cut(title+contents, minlength=0, maxlength=3, jamo_delete=True)
word_to_idx, idx_to_word = convert_dic(title + contents)

def get_word_idx(batch):
    batch_list = []
    for sentence in batch:
        sentence_list = []
        for word in sentence:
            sentence_list.append(word_to_idx.get(word))
        batch_list.append(sentence_list)
    return batch_list

def feed_dict(train):
    if train == True:
        batch_idx = np.random.choice(len(title), 2, False)
        en_ = tf.one_hot(get_word_idx(contents[batch_idx]), depth=encoder_size)
        de_ = tf.one_hot(get_word_idx(title[batch_idx]), depth=decoder_size)
        y_ = tf.one_hot(get_word_idx(title[batch_idx]), depth=decoder_size)
        return {encoder_inputs: en_, decoder_inputs: de_, target: y_}
    else:
        batch_idx = np.random.choice(len(title), 2, False)
        en_ = tf.one_hot(get_word_idx(contents[batch_idx]), depth=encoder_size)
        de_ = tf.one_hot(get_word_idx(title[batch_idx]), depth=decoder_size)
        y_ = tf.one_hot(get_word_idx(title[batch_idx]), depth=decoder_size)
        return {encoder_inputs: en_, decoder_inputs: de_, target: y_}

# parameters
forward_only = False ##training False / testing True
hidden_size = 300
vocab_size = len(idx_to_word)
num_layers = 3
learning_rate = 0.001
batch_size = 16
encoder_size = 10
# decoder_size = tool.check_doclength(title,sep=True) # (Maximum) number of time steps in this batch
decoder_size = 5
steps_per_checkpoint = 10

source_vocab_size = vocab_size
target_vocab_size = vocab_size
batch_size = batch_size
encoder_size = encoder_size
decoder_size = decoder_size
learning_rate = tf.Variable(float(learning_rate), trainable=False, name="learning_rate")
global_step = tf.Variable(0, trainable=False, name="global_step")

# networks
W = tf.Variable(tf.random_normal([hidden_size, vocab_size]), name="output_to_hidden_weight")
b = tf.Variable(tf.random_normal([vocab_size]), name="output_to_hidden_bias")
output_projection = (W, b)

encoder_inputs = [tf.placeholder(tf.float32, [batch_size, source_vocab_size], name="encoder_inputs" + str(_)) for _ in range(encoder_size)]  # 인덱스만 있는 데이터 (원핫 인코딩 미시행)
decoder_inputs = [tf.placeholder(tf.float32, [batch_size, target_vocab_size], name="decoder_inputs" + str(_)) for _ in range(decoder_size)]
targets = [tf.placeholder(tf.float32, [batch_size, target_vocab_size], name="targets" + str(_)) for _ in range(decoder_size)]
# target_weights = [tf.placeholder(tf.float32, [batch_size], name="target_weight" + str(_)) for _ in range(decoder_size)]

single_cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size)

outputs, states = tf.contrib.legacy_seq2seq.basic_rnn_seq2seq(encoder_inputs, decoder_inputs, single_cell)
logits = [tf.matmul(output_element, output_projection[0]) + output_projection[1] for output_element in outputs]
loss = []

# for logit, target, target_weight in zip(logits, targets, target_weights):
for logit, target in zip(logits, targets):
    ce_loss = tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=logit)
    # ce_loss = tf.nn.weighted_cross_entropy_with_logits(targets=target, logits=logit, pos_weight=target_weight)
    loss.append(ce_loss)
    # loss.append(ce_loss * target_weight)
cost = tf.add_n(loss)
# cost = tf.reduce_mean(tf.add_n(loss))
train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# loss = tf.contrib.legacy_seq2seq.sequence_loss(logits=logits, targets=targets, weights=target_weights)

sess = tf.Session()
train_writer = tf.summary.FileWriter("C:/Users/user/Desktop/tmp", sess.graph)
sess.run(tf.global_variables_initializer())


sess.run([train_op, cost], feed_dict(True))


####################  전부 수정   ####################
# import os
# import sys
# sys.path.append(os.getcwd()+"/Text Summarization/")
# import tool
import numpy as np
import pandas as pd
import tensorflow as tf

# data loading
data_path = 'C:/Users/user/Desktop/2016newscorpus2.csv'

def loading_data(data_path):
    # R에서 title과 contents만 csv로 저장한걸 불러와서 제목과 컨텐츠로 분리
    # write.csv(corpus, data_path, fileEncoding='utf-8', row.names=F)
    corpus = np.array(pd.read_table(data_path, sep=",", encoding="utf-8"))
    title = []
    body = []
    for idx, doc in enumerate(corpus):
        title.append(doc[0].split())
        body.append(doc[1].split())
        if idx % 100000 == 0:
            print('%d docs' % (idx))
    return np.asarray(title), np.asarray(body)

title, contents = loading_data(data_path)
input = title + contents

def convert_dic(input):
    uniq_word = []
    for i in range(len(input)):
        uniq_word = list(set(uniq_word+input[i]))
    uniq_word.sort()
    idx = list(range(len(uniq_word)))
    word_to_idx = dict(zip(uniq_word, idx))
    idx_to_word = dict(zip(idx, uniq_word))
    return word_to_idx, idx_to_word

# word_to_ix, ix_to_word = tool.make_dict_all_cut(title+contents, minlength=0, maxlength=3, jamo_delete=True)
word_to_idx, idx_to_word = convert_dic(title + contents)

def get_word_idx(batch, input_length):
    batch_list = []
    for sentence in batch:
        sentence_list = []
        for word in sentence:
            sentence_list.append(word_to_idx.get(word))
        if len(sentence_list) > input_length:
            sentence_list = sentence_list[:input_length]
        elif len(sentence_list) < input_length:

        batch_list.append(sentence_list)
    return batch_list

def feed_dict(train):
    if train == True:
        batch_idx = np.random.choice(len(title), 2, False)
        en_ = tf.one_hot(np.asarray(get_word_idx(contents[batch_idx], encoder_size)), depth=vocab_size)
        de_ = tf.one_hot(np.asarray(get_word_idx(title[batch_idx], decoder_size)), depth=vocab_size)
        y_ = tf.one_hot(np.asarray(get_word_idx(title[batch_idx], decoder_size)), depth=vocab_size)
        return {encoder_inputs: en_, decoder_inputs: de_, target: y_}
        sess.run(en_)
    # else:
    #     batch_idx = np.random.choice(len(title), 2, False)
    #     en_ = tf.one_hot(get_word_idx(contents[batch_idx]), depth=encoder_size)
    #     de_ = tf.one_hot(get_word_idx(title[batch_idx]), depth=decoder_size)
    #     y_ = tf.one_hot(get_word_idx(title[batch_idx]), depth=decoder_size)
    #     return {encoder_inputs: en_, decoder_inputs: de_, target: y_}

# parameters
forward_only = False ##training False / testing True
hidden_size = 300
vocab_size = len(idx_to_word)
num_layers = 3
learning_rate = 0.001
batch_size = 16
encoder_size = 10
# decoder_size = tool.check_doclength(title,sep=True) # (Maximum) number of time steps in this batch
decoder_size = 5
steps_per_checkpoint = 10

source_vocab_size = vocab_size
target_vocab_size = vocab_size
batch_size = batch_size
encoder_size = encoder_size
decoder_size = decoder_size
learning_rate = tf.Variable(float(learning_rate), trainable=False, name="learning_rate")
global_step = tf.Variable(0, trainable=False, name="global_step")

# networks
W = tf.Variable(tf.random_normal([hidden_size, vocab_size]), name="output_to_hidden_weight")
b = tf.Variable(tf.random_normal([vocab_size]), name="output_to_hidden_bias")
output_projection = (W, b)

encoder_inputs = [tf.placeholder(tf.float32, [batch_size, source_vocab_size], name="encoder_inputs" + str(_)) for _ in range(encoder_size)]  # 인덱스만 있는 데이터 (원핫 인코딩 미시행)
decoder_inputs = [tf.placeholder(tf.float32, [batch_size, target_vocab_size], name="decoder_inputs" + str(_)) for _ in range(decoder_size)]
targets = [tf.placeholder(tf.float32, [batch_size, target_vocab_size], name="targets" + str(_)) for _ in range(decoder_size)]
# target_weights = [tf.placeholder(tf.float32, [batch_size], name="target_weight" + str(_)) for _ in range(decoder_size)]

single_cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size)

outputs, states = tf.contrib.legacy_seq2seq.basic_rnn_seq2seq(encoder_inputs, decoder_inputs, single_cell)
logits = [tf.matmul(output_element, output_projection[0]) + output_projection[1] for output_element in outputs]
loss = []

# for logit, target, target_weight in zip(logits, targets, target_weights):
for logit, target in zip(logits, targets):
    ce_loss = tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=logit)
    # ce_loss = tf.nn.weighted_cross_entropy_with_logits(targets=target, logits=logit, pos_weight=target_weight)
    loss.append(ce_loss)
    # loss.append(ce_loss * target_weight)
cost = tf.add_n(loss)
# cost = tf.reduce_mean(tf.add_n(loss))
train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# loss = tf.contrib.legacy_seq2seq.sequence_loss(logits=logits, targets=targets, weights=target_weights)

sess = tf.Session()
train_writer = tf.summary.FileWriter("C:/Users/user/Desktop/tmp", sess.graph)
sess.run(tf.global_variables_initializer())


sess.run([train_op, cost], feed_dict(True))
