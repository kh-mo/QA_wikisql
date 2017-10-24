import numpy as np
import pandas as pd
import re
import tensorflow as tf
import random
import pickle
from collections import defaultdict
import operator

####################################################
# get train idx function                           #
####################################################
def get_train_idx(data_length, train_prop=0.9):
    random.seed(1234)
    idx = np.random.permutation(np.arange(data_length))
    train_idx = idx[:round(train_prop * data_length)]
    test_idx = idx[-(data_length-round(train_prop * data_length)):]
    return train_idx, test_idx

####################################################
# cut words function                               #
####################################################
def cut(contents, cut=2):
    results = []
    for idx, content in enumerate(contents):
        words = content.split()
        result = []
        for word in words:
            result.append(word[:cut])
        results.append(' '.join([token for token in result]))
    return results

####################################################
# divide raw train/test set function               #
####################################################
def divide(x, y, train_prop):
    random.seed(1234)
    x = np.array(x)
    y = np.array(y)
    #corpus = np.array(corpus)
    tmp = np.random.permutation(np.arange(len(x)))
    x_tr = x[tmp][:round(train_prop * len(x))]
    #corpus_tr = corpus[tmp][:round(train_prop * len(x))]
    y_tr = y[tmp][:round(train_prop * len(x))]
    x_te = x[tmp][-(len(x)-round(train_prop * len(x))):]
    y_te = y[tmp][-(len(x)-round(train_prop * len(x))):]
    return x_tr, x_te, y_tr, y_te, tmp[:round(train_prop * len(x))]

####################################################
# batch function                                   #
####################################################
def get_batch(data, batch_size, num_epochs, data_idx, word2vec, max_document_length, word2vec_model):
    contents, points = zip(*data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            indexes = data_idx[start_index:end_index]
            batch_contents = []
            batch_points = []
            for index in indexes:
                batch_contents.append(contents[index])
                batch_points.append(points[index])
            if word2vec:
                result_contents = make_word2vec_input(np.array(batch_contents), max_document_length, word2vec_model)
                result_points = make_output(np.array(batch_points))
                yield list(zip(result_contents, result_points))
            else:
                yield data[start_index:end_index]

####################################################
# making word2vec input function                   #
####################################################
def load_word2vec(word2vec_path):
    with open(word2vec_path, 'rb') as f:
        [embed_model] = pickle.load(f)
    return embed_model

def make_word2vec_input(documents, max_document_length, embed_model):
    results = []
    for document in documents:
        result = np.zeros((max_document_length, embed_model.vector_size))
        words = document.split()[:max_document_length]
        for word_idx, word in enumerate(words):
            if word in embed_model.vocab:
                result[word_idx] = embed_model[word]
        results.append(result)
    return results

####################################################
# making raw input function                        #
####################################################
def make_raw_input(documents, max_document_length):
    # tensorflow.contrib.learn.preprocessing 내에 VocabularyProcessor라는 클래스를 이용
    # 모든 문서에 등장하는 단어들에 인덱스를 할당
    # 길이가 다른 문서를 max_document_length로 맞춰주는 역할
    vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(max_document_length)  # 객체 선언
    x = np.array(list(vocab_processor.fit_transform(documents)))
    ### 텐서플로우 vocabulary processor
    # Extract word:id mapping from the object.
    # word to ix 와 유사
    vocab_dict = vocab_processor.vocabulary_._mapping
    # Sort the vocabulary dictionary on the basis of values(id).
    sorted_vocab = sorted(vocab_dict.items(), key=lambda x: x[1])
    # Treat the id's as index into list and create a list of words in the ascending order of id's
    # word with id i goes at index i of the list.
    vocabulary = list(list(zip(*sorted_vocab))[0])
    return x, vocabulary, len(vocab_processor.vocabulary_), vocab_processor

####################################################
# make output function                             #
####################################################
def make_output(points, threshold=2.5):
    results = np.zeros((len(points),2))
    for idx, point in enumerate(points):
        if point > threshold:
            results[idx,0] = 1
        else:
            results[idx,1] = 1
    return results

####################################################
# check maxlength function                         #
####################################################
def check_maxlength(contents):
    max_document_length = 0
    for document in contents:
        document_length = len(document.split())
        if document_length > max_document_length:
            max_document_length = document_length
    return max_document_length

####################################################
# loading function                                 #
####################################################
def loading_rdata(data_path, minlength=30, eng=True, num=True, punc=False):
    # R에서 title과 contents만 csv로 저장한걸 불러와서 제목과 컨텐츠로 분리
    # write.csv(corpus, data_path, fileEncoding='utf-8', row.names=F)
    corpus = pd.read_table(data_path, sep=",", encoding="utf-8")
    corpus = np.array(corpus)
    contents = []
    points = []
    for idx,doc in enumerate(corpus):
        if isNumber(doc[0]) is False and len(doc[0].split()) > minlength:
            content = normalize(doc[0], english=eng, number=num, punctuation=punc)
            contents.append(content)
            points.append(doc[1])
        if idx % 100000 is 0:
            print('%d docs / %d save' % (idx, len(contents)))
    return contents, points

def isNumber(s):
  try:
    float(s)
    return True
  except ValueError:
    return False


####################################################
# tokenizing function                              #
####################################################
from collections import defaultdict
import math
import sys



class CohesionProbability:
    def __init__(self, left_min_length=1, left_max_length=10, right_min_length=1, right_max_length=6):

        self.left_min_length = left_min_length
        self.left_max_length = left_max_length
        self.right_min_length = right_min_length
        self.right_max_length = right_max_length

        self.L = defaultdict(int)
        self.R = defaultdict(int)

    def get_cohesion_probability(self, word):

        if not word:
            return (0, 0, 0, 0)

        word_len = len(word)

        l_freq = 0 if not word in self.L else self.L[word]
        r_freq = 0 if not word in self.R else self.R[word]

        if word_len == 1:
            return (0, 0, l_freq, r_freq)

        l_cohesion = 0
        r_cohesion = 0

        # forward cohesion probability (L)
        if (self.left_min_length <= word_len) and (word_len <= self.left_max_length):

            l_sub = word[:self.left_min_length]
            l_sub_freq = 0 if not l_sub in self.L else self.L[l_sub]

            if l_sub_freq > 0:
                l_cohesion = np.power((l_freq / float(l_sub_freq)), (1 / (word_len - len(l_sub) + 1.0)))

        # backward cohesion probability (R)
        if (self.right_min_length <= word_len) and (word_len <= self.right_max_length):

            r_sub = word[-1 * self.right_min_length:]
            r_sub_freq = 0 if not r_sub in self.R else self.R[r_sub]

            if r_sub_freq > 0:
                r_cohesion = np.power((r_freq / float(r_sub_freq)), (1 / (word_len - len(r_sub) + 1.0)))

        return (l_cohesion, r_cohesion, l_freq, r_freq)

    def get_all_cohesion_probabilities(self):

        cp = {}
        words = set(self.L.keys())
        for word in self.R.keys():
            words.add(word)

        for word in words:
            cp[word] = self.get_cohesion_probability(word)

        return cp

    def counter_size(self):
        return (len(self.L), len(self.R))

    def prune_extreme_case(self, min_count):

        before_size = self.counter_size()
        self.L = defaultdict(int, {k: v for k, v in self.L.items() if v > min_count})
        self.R = defaultdict(int, {k: v for k, v in self.R.items() if v > min_count})
        after_size = self.counter_size()

        return (before_size, after_size)

    def train(self, sents, num_for_pruning=0, min_count=5):

        for num_sent, sent in enumerate(sents):
            for word in sent.split():

                if not word:
                    continue

                word_len = len(word)

                for i in range(self.left_min_length, min(self.left_max_length, word_len) + 1):
                    self.L[word[:i]] += 1

                # for i in range(self.right_min_length, min(self.right_max_length, word_len)+1):
                for i in range(self.right_min_length, min(self.right_max_length, word_len)):
                    self.R[word[-i:]] += 1

            if (num_for_pruning > 0) and ((num_sent + 1) % num_for_pruning == 0):
                self.prune_extreme_case(min_count)

        if (num_for_pruning > 0) and ((num_sent + 1) % num_for_pruning == 0):
            self.prune_extreme_case(min_count)

    def extract(self, min_count=5, min_cohesion=(0.05, 0), min_droprate=0.8, remove_subword=True):

        word_to_score = self.get_all_cohesion_probabilities()
        word_to_score = {word: score for word, score in word_to_score.items()
                         if (score[0] >= min_cohesion[0])
                         and (score[1] >= min_cohesion[1])
                         and (score[2] >= min_count)}

        if not remove_subword:
            return word_to_score

        words = {}

        for word, score in sorted(word_to_score.items(), key=lambda x: len(x[0])):
            len_word = len(word)
            if len_word <= 2:
                words[word] = score
                continue

            try:
                subword = word[:-1]
                subscore = self.get_cohesion_probability(subword)
                droprate = score[2] / subscore[2]

                if (droprate >= min_droprate) and (subword in words):
                    del words[subword]

                words[word] = score

            except:
                print(word, score, subscore)
                break

        return words

    def transform(self, docs, l_word_set):

        def left_match(word):
            for i in reversed(range(1, len(word) + 1)):
                if word[:i] in l_word_set:
                    return word[:i]
            return ''

        return [[left_match(word) for sent in doc.split('  ') for word in sent.split() if left_match(word)] for doc in
                docs]

    def load(self, fname):
        try:
            with open(fname, encoding='utf-8') as f:

                next(f)  # SKIP: parameters(left_min_length left_max_length ...
                token = next(f).split()
                self.left_min_length = int(token[0])
                self.left_max_length = int(token[1])
                self.right_min_length = int(token[2])
                self.right_max_length = int(token[3])

                next(f)  # SKIP: L count
                is_right_side = False

                for line in f:

                    if '# R count' in line:
                        is_right_side = True
                        continue

                    token = line.split('\t')
                    if is_right_side:
                        self.R[token[0]] = int(token[1])
                    else:
                        self.L[token[0]] = int(token[1])

        except Exception as e:
            print(e)

    def save(self, fname):
        try:
            with open(fname, 'w', encoding='utf-8') as f:

                f.write('# parameters(left_min_length left_max_length right_min_length right_max_length)\n')
                f.write('%d %d %d %d\n' % (
                self.left_min_length, self.left_max_length, self.right_min_length, self.right_max_length))

                f.write('# L count')
                for word, freq in self.L.items():
                    f.write('%s\t%d\n' % (word, freq))

                f.write('# R count')
                for word, freq in self.R.items():
                    f.write('%s\t%d\n' % (word, freq))

        except Exception as e:
            print(e)

    def words(self):
        words = set(self.L.keys())
        words = words.union(set(self.R.keys()))
        return words


class BranchingEntropy:
    def __init__(self, min_length=2, max_length=7):

        self.min_length = min_length
        self.max_length = max_length

        self.encoder = IntegerEncoder()

        self.L = defaultdict(lambda: defaultdict(int))
        self.R = defaultdict(lambda: defaultdict(int))

    def get_all_access_variety(self):

        av = {}
        words = set(self.L.keys())
        words += set(self.R.keys())

        for word in words:
            av[word] = self.get_access_variety(word)

        return av

    def get_access_variety(self, word, ignore_space=False):

        return (len(self.get_left_branch(word, ignore_space)), len(self.get_right_branch(word, ignore_space)))

    def get_all_branching_entropies(self, ignore_space=False):

        be = {}
        words = set(self.L.keys())
        for word in self.R.keys():
            words.add(word)

        for word in words:
            be[self.encoder.decode(word)] = self.get_branching_entropy(word, ignore_space)

        return be

    def get_branching_entropy(self, word, ignore_space=False):

        be_l = self.entropy(self.get_left_branch(word, ignore_space))
        be_r = self.entropy(self.get_right_branch(word, ignore_space))
        return (be_l, be_r)

    def entropy(self, dic):

        if not dic:
            return 0.0

        sum_count = sum(dic.values())
        entropy = 0

        for freq in dic.values():
            prob = freq / sum_count
            entropy += prob * math.log(prob)

        return -1 * entropy

    def get_left_branch(self, word, ignore_space=False):

        if isinstance(word, int):
            word_index = word
        else:
            word_index = self.encoder.encode(word)

        if (word_index == -1) or (not word_index in self.L):
            return {}

        branch = self.L[word_index]

        if ignore_space:
            return {w: f for w, f in branch.items() if not ' ' in self.encoder.decode(w, unknown=' ')}
        else:
            return branch

    def get_right_branch(self, word, ignore_space=False):

        if isinstance(word, int):
            word_index = word
        else:
            word_index = self.encoder.encode(word)

        if (word_index == -1) or (not word_index in self.R):
            return {}

        branch = self.R[word_index]

        if ignore_space:
            return {w: f for w, f in branch.items() if not ' ' in self.encoder.decode(w, unknown=' ')}
        else:
            return branch

    def counter_size(self):
        return (len(self.L), len(self.R))

    def prune_extreme_case(self, min_count):

        # TODO: encoder remove & compatify
        before_size = self.counter_size()
        self.L = defaultdict(lambda: defaultdict(int),
                             {word: dic for word, dic in self.L.items() if sum(dic.values()) > min_count})
        self.R = defaultdict(lambda: defaultdict(int),
                             {word: dic for word, dic in self.R.items() if sum(dic.values()) > min_count})
        after_size = self.counter_size()

        return (before_size, after_size)

    def train(self, sents, min_count=5, num_for_pruning=10000):

        for num_sent, sent in enumerate(sents):

            sent = sent.strip()
            if not sent:
                continue

            sent = ' ' + sent.strip() + ' '
            length = len(sent)

            for i in range(1, length - 1):
                for window in range(self.min_length, self.max_length + 1):

                    if i + window - 1 >= length:
                        continue

                    word = sent[i:i + window]
                    if ' ' in word:
                        continue

                    word_index = self.encoder.fit(word)

                    if sent[i - 1] == ' ':
                        left_extension = sent[max(0, i - 2):i + window]
                    else:
                        left_extension = sent[i - 1:i + window]

                    if sent[i + window] == ' ':
                        right_extension = sent[i:min(length, i + window + 2)]
                    else:
                        right_extension = sent[i:i + window + 1]

                    if left_extension == None or right_extension == None:
                        print(sent, i, window)

                    left_index = self.encoder.fit(left_extension)
                    right_index = self.encoder.fit(right_extension)

                    self.L[word_index][left_index] += 1
                    self.R[word_index][right_index] += 1

            if (num_for_pruning > 0) and ((num_sent + 1) % num_for_pruning == 0):
                before, after = self.prune_extreme_case(min_count)
                sys.stdout.write('\rnum sent = %d: %s --> %s' % (num_sent, str(before), str(after)))

        if (num_for_pruning > 0) and ((num_sent + 1) % num_for_pruning == 0):
            self.prune_extreme_case(min_count)
            sys.stdout.write('\rnum_sent = %d: %s --> %s' % (num_sent, str(before), str(after)))

    def load(self, model_fname, encoder_fname):

        self.encoder.load(encoder_fname)

        try:
            with open(model_fname, encoding='utf-8') as f:

                next(f)  # SKIP: parameters (min_length, max_length)
                token = next(f).split()
                self.min_length = int(token[0])
                self.max_length = int(token[1])

                next(f)  # SKIP: left side extension
                is_right_side = True

                for line in f:

                    if '# right side extension' in line:
                        is_right_side = True
                        continue

                    token = line.split();
                    word = int(token[0])
                    extension = int(token[1])
                    freq = int(token[2])

                    if is_right_side:
                        self.R[word][extension] = freq
                    else:
                        self.L[word][extension] = freq

        except Exception as e:
            print(e)

    def save(self, model_fname, encoder_fname):

        self.encoder.save(encoder_fname)

        try:
            with open(model_fname, 'w', encoding='utf-8') as f:

                f.write("# parameters (min_length max_length)\n")
                f.write('%d %d\n' % (self.min_length, self.max_length))

                f.write('# left side extension\n')
                for word, extension_dict in self.L.items():
                    for extension, freq in extension_dict.items():
                        f.write('%d %d %d\n' % (word, extension, freq))

                f.write('# right side extension\n')
                for word, extension_dict in self.R.items():
                    for extension, freq in extension_dict.items():
                        f.write('%d %d %d\n' % (word, extension, freq))

        except Exception as e:
            print(e)

    def words(self):
        return set(self.encoder.inverse)


class KR_WordRank:
    """Unsupervised Korean Keyword Extractor
    Implementation of Kim, H. J., Cho, S., & Kang, P. (2014). KR-WordRank:
    An Unsupervised Korean Word Extraction Method Based on WordRank.
    Journal of Korean Institute of Industrial Engineers, 40(1), 18-33.
    """

    def __init__(self, min_count=5, max_length=10):
        self.min_count = min_count
        self.max_length = max_length
        self.sum_weight = 1
        self.vocabulary = {}
        self.index2vocab = []

    def scan_vocabs(self, docs, verbose=True):
        self.vocabulary = {}
        if verbose:
            print('scan vocabs ... ')

        counter = {}
        for doc in docs:

            for token in doc.split():
                len_token = len(token)
                counter[(token, 'L')] = counter.get((token, 'L'), 0) + 1

                for e in range(1, min(len(token), self.max_length)):
                    if (len_token - e) > self.max_length:
                        continue

                    l_sub = (token[:e], 'L')
                    r_sub = (token[e:], 'R')
                    counter[l_sub] = counter.get(l_sub, 0) + 1
                    counter[r_sub] = counter.get(r_sub, 0) + 1

        counter = {token: freq for token, freq in counter.items() if freq >= self.min_count}
        for token, _ in sorted(counter.items(), key=lambda x: x[1], reverse=True):
            self.vocabulary[token] = len(self.vocabulary)

        self._build_index2vocab()

        if verbose:
            print('num vocabs = %d' % len(counter))
        return counter

    def _build_index2vocab(self):
        self.index2vocab = [vocab for vocab, index in sorted(self.vocabulary.items(), key=lambda x: x[1])]
        self.sum_weight = len(self.index2vocab)

    def extract(self, docs, beta=0.85, max_iter=10, verbose=True, vocabulary={}, bias={}, rset={}):
        rank, graph = self.train(docs, beta, max_iter, verbose, vocabulary, bias)

        lset = {self.int2token(idx)[0]: r for idx, r in rank.items() if self.int2token(idx)[1] == 'L'}
        if not rset:
            rset = {self.int2token(idx)[0]: r for idx, r in rank.items() if self.int2token(idx)[1] == 'R'}

        keywords = self._select_keywords(lset, rset)
        keywords = self._filter_compounds(keywords)
        keywords = self._filter_subtokens(keywords)

        return keywords, rank, graph

    def _select_keywords(self, lset, rset):
        keywords = {}
        for word, r in sorted(lset.items(), key=lambda x: x[1], reverse=True):
            len_word = len(word)
            if len_word == 1:
                continue

            is_compound = False
            for e in range(2, len_word):
                if (word[:e] in keywords) and (word[:e] in rset):
                    is_compound = True
                    break

            if not is_compound:
                keywords[word] = r

        return keywords

    def _filter_compounds(self, keywords):
        keywords_ = {}
        for word, r in sorted(keywords.items(), key=lambda x: x[1], reverse=True):
            len_word = len(word)

            if len_word <= 2:
                keywords_[word] = r
                continue

            if len_word == 3:
                if word[:2] in keywords_:
                    continue

            is_compound = False
            for e in range(2, len_word - 1):
                if (word[:e] in keywords) and (word[:e] in keywords):
                    is_compound = True
                    break

            if not is_compound:
                keywords_[word] = r

        return keywords_

    def _filter_subtokens(self, keywords):
        subtokens = set()
        keywords_ = {}

        for word, r in sorted(keywords.items(), key=lambda x: x[1], reverse=True):
            subs = {word[:e] for e in range(2, len(word) + 1)}

            is_subtoken = False
            for sub in subs:
                if sub in subtokens:
                    is_subtoken = True
                    break

            if not is_subtoken:
                keywords_[word] = r
                subtokens.update(subs)

        return keywords_

    def train(self, docs, beta=0.85, max_iter=10, verbose=True, vocabulary={}, bias={}):
        if (not vocabulary) and (not self.vocabulary):
            self.scan_vocabs(docs, verbose)
        elif (not vocabulary):
            self.vocabulary = vocabulary
            self._build_index2vocab()

        graph = self._construct_word_graph(docs)

        dw = self.sum_weight / len(self.vocabulary)
        rank = {node: dw for node in graph.keys()}

        for num_iter in range(1, max_iter + 1):
            rank = self._update(rank, graph, bias, dw, beta)
            sys.stdout.write('\riter = %d' % num_iter)
        print('\rdone')

        return rank, graph

    def token2int(self, token):
        return self.vocabulary.get(token, -1)

    def int2token(self, index):
        return self.index2vocab[index] if (0 <= index < len(self.index2vocab)) else None

    def _construct_word_graph(self, docs):
        def normalize(graph):
            graph_ = defaultdict(lambda: defaultdict(lambda: 0))
            for from_, to_dict in graph.items():
                sum_ = sum(to_dict.values())
                for to_, w in to_dict.items():
                    graph_[to_][from_] = w / sum_
            return graph_

        graph = defaultdict(lambda: defaultdict(lambda: 0))
        for doc in docs:

            tokens = doc.split()

            if not tokens:
                continue

            links = []
            for token in tokens:
                links += self._intra_link(token)

            if len(tokens) > 1:
                tokens = [tokens[-1]] + tokens + [tokens[0]]
                links += self._inter_link(tokens)

            links = self._check_token(links)
            if not links:
                continue

            links = self._encode_token(links)
            for l_node, r_node in links:
                graph[l_node][r_node] += 1
                graph[r_node][l_node] += 1

        graph = normalize(graph)
        return graph

    def _intra_link(self, token):
        links = []
        len_token = len(token)
        for e in range(1, min(len_token, 10)):
            if (len_token - e) > self.max_length:
                continue
            links.append(((token[:e], 'L'), (token[e:], 'R')))
        return links

    def _inter_link(self, tokens):
        def rsub_to_token(t_left, t_curr):
            return [((t_left[-b:], 'R'), (t_curr, 'L')) for b in range(1, min(10, len(t_left)))]

        def token_to_lsub(t_curr, t_rigt):
            return [((t_curr, 'L'), (t_rigt[:e], 'L')) for e in range(1, min(10, len(t_rigt)))]

        links = []
        for i in range(1, len(tokens) - 1):
            links += rsub_to_token(tokens[i - 1], tokens[i])
            links += token_to_lsub(tokens[i], tokens[i + 1])
        return links

    def _check_token(self, token_list):
        return [(token[0], token[1]) for token in token_list if
                (token[0] in self.vocabulary and token[1] in self.vocabulary)]

    def _encode_token(self, token_list):
        return [(self.vocabulary[token[0]], self.vocabulary[token[1]]) for token in token_list]

    def _update(self, rank, graph, bias, dw, beta):
        rank_new = {}
        for to_node, from_dict in graph.items():
            rank_new[to_node] = sum([w * rank[from_node] for from_node, w in from_dict.items()])
            rank_new[to_node] = beta * rank_new[to_node] + (1 - beta) * bias.get(to_node, dw)
        return rank_new


class IntegerEncoder:
    def __init__(self):

        self.mapper = {}
        self.inverse = []
        self.num_object = 0

    def compatify(self):

        fixer = {}
        pull_index = 0
        none_index = []

        for i, x in enumerate(self.inverse):
            if x == None:
                none_index.append(i)
                pull_index += 1
            elif pull_index > 0:
                fixed = i - pull_index
                fixer[i] = fixed
                self.mapper[x] = fixed

        for i in reversed(none_index):
            del self.inverse[i]

        return fixer

    def __getitem__(self, x):
        if type(x) == int:
            if x < self.num_object:
                return self.inverse[x]
            else:
                return None
        if x in self.mapper:
            return self.mapper[x]
        else:
            return -1

    def decode(self, i, unknown=None):
        if i >= 0 and i < self.num_object:
            return self.inverse[i]
        else:
            return unknown

    def encode(self, x, unknown=-1):
        if x in self.mapper:
            return self.mapper[x]
        else:
            return unknown

    def fit(self, x):
        if x in self.mapper:
            return self.mapper[x]
        else:
            self.mapper[x] = self.num_object
            self.num_object += 1
            self.inverse.append(x)
            return (self.num_object - 1)

    def keys(self):
        return self.inverse

    def remove(self, x):
        if x in self.mapper:
            i = self.mapper[x]
            del self.mapper[x]
            self.inverse[i] = None
            self.num_object -= 1

    def save(self, fname, to_str=lambda x: str(x)):
        try:
            with open(fname, 'w', encoding='utf-8') as f:
                for x in self.inverse:
                    f.write('%s\n' % to_str(x))
        except Exception as e:
            print(e)

    def load(self, fname, parse=lambda x: x.replace('\n', '')):
        try:
            with open(fname, encoding='utf-8') as f:
                for line in f:
                    x = parse(line)
                    self.inverse.append(x)
                    self.mapper[x] = self.num_object
                    self.num_object += 1
        except Exception as e:
            print(e)
            print('line number = %d' % self.num_object)

    def __len__(self):
        return self.num_object


class RegexTokenizer:
    def __init__(self):
        self.patterns = [
            ('number', re.compile('[-+]?\d*[\.]?[\d]+|[-+]?\d+')),
            ('korean', re.compile('[가-힣]+')),
            ('jaum', re.compile('[ㄱ-ㅎ]+')),
            ('moum', re.compile('[ㅏ-ㅣ]+')),
            ('english & latin', re.compile("[a-zA-ZÀ-ÿ]+[[`']?s]*|[a-zA-ZÀ-ÿ]+"))
        ]

        self.doublewhite_pattern = re.compile('\s+')

    def tokenize(self, s, debug=False):
        '''
        Usage
        s = "이거에서+3.12같은34숫자나-1.2like float해해 같은aÀÿfafAis`s-1찾아서3.1.2.1해ㅋㅋㅜㅠ봐 Bob`s job.1"
        tokenizer = RegularTokenizer()
        tokenizer.tokenize(s)
        [['이거에서', '+3.12', '같은', '34', '숫자나', '-1.2', 'like'],
         ['float', '해해'],
         ['같은', 'aÀÿfafAis`s', '-1', '찾아서', '3.1', '.2', '.1', '해', 'ㅋㅋ', 'ㅜㅠ', '봐'],
         ['Bob`s'],
         ['job', '.1']]
        '''
        return [self._tokenize(t, debug) for t in s.split()]

    def _tokenize(self, s, debug=False):
        for name, pattern in self.patterns:

            founds = pattern.findall(s)
            if not founds:
                continue

            if debug:
                print('\n%s' % name)
                print(founds)

            found = founds.pop(0)
            len_found = len(found)

            s_ = ''
            b = 0
            for i, c in enumerate(s):

                if b > i:
                    continue

                if s[i:i + len_found] == found:
                    s_ += ' %s ' % s[i:i + len_found]
                    b = i + len_found

                    if not founds:
                        s_ += s[b:]
                        break
                    else:
                        found = founds.pop(0)
                        len_found = len(found)

                    continue
                s_ += c
            s = s_

        s = self.doublewhite_pattern.sub(' ', s).strip().split()
        # TODO: handle 3.1.2.1
        return s


class LTokenizer:
    def __init__(self, scores={}, default_score=0.0):
        self.scores = scores
        self.ds = default_score

    def tokenize(self, sentence):
        def token_to_lr(token):
            length = len(token)
            if length <= 2: return (token, '')
            candidates = [(token[:e], token[e:]) for e in range(2, length + 1)]
            candidates = [(self.scores.get(t[0], self.ds), t[0], t[1]) for t in candidates]
            best = sorted(candidates, key=lambda x: (x[0], len(x[1])), reverse=True)[0]
            return (best[1], best[2])

        return [token_to_lr(token) for token in sentence.split()]


class MaxScoreTokenizer:
    def __init__(self, max_length=10, scores={}, default_score=0.0):
        self.max_length = max_length
        self.scores = scores
        self.ds = default_score

    def tokenize(self, sentence):
        return [self._recursive_tokenize(token) for token in sentence.split()]

    def _recursive_tokenize(self, token, range_l=0, debug=False):

        length = len(token)
        if length <= 2:
            return [(token, 0, length, self.ds, length)]

        if range_l == 0:
            range_l = min(self.max_length, length)

        scores = self._initialize(token, range_l, length)
        if debug:
            pprint(scores)

        result = self._find(scores)

        adds = self._add_inter_subtokens(token, result)

        if result[-1][2] != length:
            adds += self._add_first_subtoken(token, result)

        if result[0][1] != 0:
            adds += self._add_last_subtoken(token, result)

        return sorted(result + adds, key=lambda x: x[1])

    def _initialize(self, token, range_l, length):
        scores = []
        for b in range(0, length - 1):
            for r in range(2, range_l + 1):
                e = b + r

                if e > length:
                    continue

                subtoken = token[b:e]
                score = self.scores.get(subtoken, self.ds)
                scores.append((subtoken, b, e, score, r))

        #return sorted(scores, key=lambda x: (x[3], x[4]), reverse=True)
        return sorted(scores, key=lambda x: (x[0], x[1]), reverse=True)

    def _find(self, scores):
        result = []
        num_iter = 0

        while scores:
            word, b, e, score, r = scores.pop(0)
            result.append((word, b, e, score, r))

            if not scores:
                break

            removals = []
            for i, (_1, b_, e_, _2, _3) in enumerate(scores):
                if (b_ < e and b < e_) or (b_ < e and e_ > b):
                    removals.append(i)

            for i in reversed(removals):
                del scores[i]

            num_iter += 1
            if num_iter > 100: break

        return sorted(result, key=lambda x: x[1])

    def _add_inter_subtokens(self, token, result):
        adds = []
        for i, base in enumerate(result[:-1]):
            if base[2] == result[i + 1][1]:
                continue

            b = base[2]
            e = result[i + 1][1]
            subtoken = token[b:e]
            adds.append((subtoken, b, e, self.ds, e - b))

        return adds

    def _add_first_subtoken(self, token, result):
        b = result[-1][2]
        subtoken = token[b:]
        score = self.scores.get(subtoken, self.ds)
        return [(subtoken, b, len(token), score, len(subtoken))]

    def _add_last_subtoken(self, token, result):
        e = result[0][1]
        subtoken = token[0:e]
        score = self.scores.get(subtoken, self.ds)
        return [(subtoken, 0, e, score, e)]


class CohesionTokenizer:
    def __init__(self, cohesion):
        self.cohesion = cohesion
        self.range_l = cohesion.left_max_length

    def tokenize(self, sentence, max_ngram=4, length_penalty=-0.05, ngram=False, debug=False):

        def flatten(tokens):
            return [word for token in tokens for word in token]

        tokens = [self._recursive_tokenize(token, max_ngram, length_penalty, ngram, debug) for token in
                  sentence.split()]
        words = flatten(tokens)

        if not debug:
            tokens = [word if type(word) == str else word[0] for word in words]

        return tokens

    def _recursive_tokenize(self, token, max_ngram=4, length_penalty=-0.05, ngram=False, debug=False):

        length = len(token)
        if length <= 2:
            return [token]

        range_l = min(self.range_l, length)

        scores = self._initialize(token, range_l, length)
        if debug:
            pprint(scores)

        result = self._find(scores)

        adds = self._add_inter_subtokens(token, result)

        if result[-1][2] != length:
            adds += self._add_first_subtoken(token, result)

        if result[0][1] != 0:
            adds += self._add_last_subtoken(token, result)

        result = sorted(result + adds, key=lambda x: x[1])

        if ngram:
            result = self._extract_ngram(result, max_ngram, length_penalty)

        return result

    def _initialize(self, token, range_l, length):
        scores = []
        for b in range(0, length - 1):
            for r in range(2, range_l + 1):
                e = b + r

                if e > length:
                    continue

                subtoken = token[b:e]
                score = self.cohesion.get_cohesion_probability(subtoken)
                # (subtoken, begin, end, cohesion_l, frequency_l, range)
                scores.append((subtoken, b, e, score[0], score[2], r))

        return sorted(scores, key=lambda x: (x[3], x[5]), reverse=True)

    def _find(self, scores):
        result = []
        num_iter = 0

        while scores:
            word, b, e, cp_l, freq_l, r = scores.pop(0)
            result.append((word, b, e, cp_l, freq_l, r))

            if not scores:
                break

            removals = []
            for i, (_1, b_, e_, _2, _3, _4) in enumerate(scores):
                if (b_ < e and b < e_) or (b_ < e and e_ > b):
                    removals.append(i)

            for i in reversed(removals):
                del scores[i]

            num_iter += 1
            if num_iter > 100: break

        return sorted(result, key=lambda x: x[1])

    def _add_inter_subtokens(self, token, result):
        adds = []
        for i, base in enumerate(result[:-1]):
            if base[2] == result[i + 1][1]:
                continue

            b = base[2]
            e = result[i + 1][1]
            subtoken = token[b:e]
            adds.append((subtoken, b, e, 0, self.cohesion.L.get(subtoken, 0), e - b))

        return adds

    def _add_first_subtoken(self, token, result):
        b = result[-1][2]
        subtoken = token[b:]
        score = self.cohesion.get_cohesion_probability(subtoken)
        return [(subtoken, b, len(token), score[0], score[2], len(subtoken))]

    def _add_last_subtoken(self, token, result):
        e = result[0][1]
        subtoken = token[0:e]
        score = self.cohesion.get_cohesion_probability(subtoken)
        return [(subtoken, 0, e, score[0], score[2], e)]

    def _extract_ngram(self, words, max_ngram=4, length_penalty=-0.05):

        def ngram_average_score(words):
            words = [word for word in words if len(word) > 1]
            scores = [word[3] for word in words]
            return max(0, np.mean(scores) + length_penalty * len(scores))

        length = len(words)
        scores = []

        if length <= 1:
            return words

        for word in words:
            scores.append(word)

        for b in range(0, length - 1):
            for r in range(2, max_ngram + 1):
                e = b + r

                if e > length:
                    continue

                ngram = words[b:e]
                ngram_str = ''.join([word[0] for word in ngram])
                ngram_str_ = '-'.join([word[0] for word in ngram])

                ngram_freq = self.cohesion.L.get(ngram_str, 0)
                if ngram_freq == 0:
                    continue

                base_freq = min([word[4] for word in ngram])
                ngram_score = np.power(ngram_freq / base_freq, 1 / (r - 1)) if base_freq > 0 else 0
                ngram_score -= r * length_penalty

                scores.append((ngram_str_, words[b][1], words[e - 1][2], ngram_score, ngram_freq, 0))

        scores = sorted(scores, key=lambda x: x[3], reverse=True)
        return self._find(scores)

####################################################
# text normalizing function                        #
####################################################

# normalize index
kor_begin = 44032
kor_end = 55199
jaum_begin = 12593
jaum_end = 12622
moum_begin = 12623
moum_end = 12643
doublespace_pattern = re.compile('\s+')
repeatchars_pattern = re.compile('(\w)\\1{3,}')
#title_pattern = re.compile('\[\D+\]|\[\S+\]')

#def normalize(doc, english=False, number=False, punctuation=False, title=True, remove_repeat=0):
def normalize(doc, english=False, number=False, punctuation=False, remove_repeat=0):
    if remove_repeat > 0:
        doc = repeatchars_pattern.sub('\\1' * remove_repeat, doc)

    #if title:
    #    doc = title_pattern.sub('', doc)

    f = ''

    for c in doc:
        i = ord(c)

        if (c == ' ') or (is_korean(i)) or (is_jaum(i)) or (is_moum(i)) or (english and is_english(i)) or (
            number and is_number(i)) or (punctuation and is_punctuation(i)):
            f += c
        else:
            f += ' '

    return doublespace_pattern.sub(' ', f).strip()


def is_korean(i):
    i = to_base(i)
    return (kor_begin <= i <= kor_end) or (jaum_begin <= i <= jaum_end) or (moum_begin <= i <= moum_end)

def is_number(i):
    i = to_base(i)
    return (i >= 48 and i <= 57)

def is_english(i):
    i = to_base(i)
    return (i >= 97 and i <= 122) or (i >= 65 and i <= 90)

def is_punctuation(i):
    i = to_base(i)
    return (i == 33 or i == 34 or i == 39 or i == 44 or i == 46 or i == 63 or i == 96)

def is_jaum(i):
    i = to_base(i)
    return (jaum_begin <= i <= jaum_end)

def is_moum(i):
    i = to_base(i)
    return (moum_begin <= i <= moum_end)

def to_base(c):
    if type(c) == str:
        return ord(c)
    elif type(c) == int:
        return c
    else:
        raise TypeError