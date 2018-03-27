import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

sentences = ["나 고양이 정말 좋다",
            "나 고양이 너무 싫다",
            "너 고양이 정말 좋다",
            "너 고양이 너무 싫다",
            "우리 고양이 정말 좋다",
            "우리 고양이 너무 싫다",
            "나 개 정말 좋다",
            "나 개 너무 싫다",
            "너 개 정말 좋다",
            "너 개 너무 싫다",
            "우리 개 정말 좋다",
            "우리 개 너무 싫다",
            "나 고양이 좋다",
            "나 강아지 좋다",
            "나 동물 좋다",
            "강아지 고양이 동물",
            "여자친구 고양이 강아지 좋다",
            "고양이 생선 우유 좋다",
            "강아지 생선 싫다 우유 좋다",
            "강아지 고양이 눈 좋다",
            "나 여자친구 좋다",
            "여자친구 나 싫다",
            "여자친구 나 영화 책 음악 좋다",
            "나 게임 만화 애니 좋다",
            "고양이 강아지 싫다",
            "강아지 고양이 좋다"]

class word2vec:
    def __init__(self, corpus=None, embedding_size=2, window=2, min_count=10, iteration=1000, sg=1, num_sample=5):
        self.corpus = corpus
        self.embedding_size = embedding_size
        self.window = window
        self.min_count = min_count
        self.iteration = iteration
        self.sg = sg
        self.num_sample = num_sample
        self.word_list = []
        self.num_classes = self.word_count()
        self.word_dic = {w: i for i, w in enumerate(self.word_list)}
        self.batch_x = []
        self.batch_y = []
        self.make_batch()
        self.sess = tf.Session()

        self.x = tf.placeholder(dtype=tf.int32, shape=[None], name="input")
        self.y = tf.placeholder(dtype=tf.int32, shape=[None, 1], name="output")

        self._embeddings = tf.get_variable(name='embeddings_input', shape=[self.num_classes, self.embedding_size],
                                           dtype=tf.float32, initializer=tf.truncated_normal_initializer())
        self._selected_embed = tf.nn.embedding_lookup(params=self._embeddings, ids=self.x)

        self.nce_weights = tf.Variable(tf.random_uniform([self.num_classes, self.embedding_size], -1.0, 1.0))
        self.nce_biases = tf.Variable(tf.zeros([self.num_classes]))

        self.nce_loss = tf.reduce_mean(tf.nn.nce_loss(weights=self.nce_weights, biases=self.nce_biases,
                                                      labels=self.y, inputs=self._selected_embed,
                                                      num_sampled=self.num_sample, num_classes=self.num_classes))
        self.batch_size = 20
        self.tr_loss_hist = []
        self.train()

    def word_count(self):
        for sentence in self.corpus:
            for word in sentence.split():
                self.word_list.append(word)
                self.word_list = list(set(self.word_list))
        return len(self.word_list)

    def make_batch(self):
        for sentence in self.corpus:
            tmp = sentence.split()
            for idx, word in enumerate(tmp):
                center_word = tmp[idx]
                for i in range(-self.window, self.window + 1):
                    if i == 0:
                        continue
                    if (idx + i < 0) | (idx + i >= len(tmp)):
                        continue
                    context_word = tmp[idx + i]
                    if self.sg == 1: # skip-gram
                        self.batch_x.append(self.word_dic[center_word])
                        self.batch_y.append([self.word_dic[context_word]])
                    else: # cbow
                        self.batch_x.append(self.word_dic[context_word])
                        self.batch_y.append([self.word_dic[center_word]])

    def train(self):
        global_step = tf.Variable(0, trainable=False)
        lr = tf.train.exponential_decay(1e-4, global_step, decay_steps=100, decay_rate=0.1, staircase=True)
        train_step = tf.train.AdamOptimizer(lr).minimize(self.nce_loss)
        train_batch_x = np.array(self.batch_x)
        train_batch_y = np.array(self.batch_y)
        self.sess.run(tf.global_variables_initializer())
        for iter in range(self.iteration * train_batch_x.shape[0]):
            batch_idx = np.random.choice(train_batch_x.shape[0], self.batch_size, False)
            _, loss = self.sess.run([train_step, self.nce_loss], feed_dict={self.x:train_batch_x[batch_idx],
                                                                            self.y:train_batch_y[batch_idx]})
            if iter % 300 == 0:
                print('iter : {}, loss : {}, lr : {}'.format(iter, loss, lr))
                self.tr_loss_hist.append(loss)

a = word2vec(sentences)

plt.plot(a.tr_loss_hist)
for word in a.word_list:
    tmp = a.sess.run(a._embeddings[a.word_dic[word]])
    x, y = tmp[0], tmp[1]
    plt.scatter(x, y)
    plt.annotate(word, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')


