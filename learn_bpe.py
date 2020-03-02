#-*- coding:utf-8 -*-

import os
import re
import json
import argparse
from collections import Counter, defaultdict

def set_hyperparameters():
    parser = argparse.ArgumentParser()

    # file
    parser.add_argument("--infile", default=os.path.join(os.getcwd(), "preprocess/stanford/train_s.jsonl"))
    parser.add_argument("--rule_outfile", default=os.path.join(os.getcwd(), "preprocess/bpe_util_data/rules.txt"))
    parser.add_argument("--glossaries_outfile", default=os.path.join(os.getcwd(), "preprocess/bpe_util_data/glossaries.txt"))

    # option
    parser.add_argument("--num_symbols", type=int, default=10000000)
    parser.add_argument("--min_frequency", type=int, default=2)
    parser.add_argument("--verbose", type=str, default="false")
    args = parser.parse_args()
    return args

def learn_bpe(args):
    infile = open(args.infile, "r", encoding="utf-8")
    output = []

    # {단어 : count} 형태의 딕셔너리 반환합니다.
    vocab = get_vocabulary(infile)
    # 단어를 음절별로 나누기, 단어의 마지막 음절에는 </w>를 붙입니다.
    vocab = dict([(tuple(x[:-1]) + (x[-1] + "</w>",), y) for (x, y) in vocab.items()])
    # 단어를 count 기준으로 내림차순 정렬합니다.
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1], reverse=True)

    # stats : {[token1, token2]:count}, indices {[token1,token2]:{1번단어:count, 2번단어:count}}
    # stats는 특정 단어 페어가 등장한 총 수, indices는 해당 단어 페어가 등장한 위치
    stats, indices = get_pair_statistics(sorted_vocab)

    for i in range(args.num_symbols):
        # 가장 빈도가 높은 pair 획득
        if stats:
            most_frequent = max(stats, key=lambda x: stats[x])

        # 모든 pair들이 최소 빈도수 이하일 경우 종료
        if stats[most_frequent] < args.min_frequency:
            print('no pair has frequency >= {0}. Stopping\n'.format(args.min_frequency))
            break

        # 진행상황 보여주기
        if args.verbose == "true":
            print('pair {0}: {1} {2} -> {1}{2} (frequency {3})\n'.format(i, most_frequent[0], most_frequent[1], stats[most_frequent]))

        output.append(most_frequent)
        changes = replace_pair(most_frequent, sorted_vocab, indices)
        update_pair_statistics(most_frequent, changes, stats, indices)

    infile.close()

    return output

def get_vocabulary(file):
    vocab = Counter()
    for line in file:
        doc = json.loads(line)
        for word in doc['nl_s'].strip("\r\n ").split(' '):
            if word:
                vocab[word] += 1
        for col in doc['col_s']:
            for word in col.strip("\r\n ").split(' '):
                if word:
                    vocab[word] += 1
    return vocab

def get_pair_statistics(file):
    stats = defaultdict(int)
    # dict 안의 dict, {key : {key : value}} 구조, {(prev_char, char) : {1 : 1}}
    indices = defaultdict(lambda: defaultdict(int))

    for i, (word, freq) in enumerate(file):
        prev_char = word[0]
        for char in word[1:]:
            stats[prev_char, char] += freq
            indices[prev_char, char][i] += 1
            prev_char = char
    return stats, indices

def replace_pair(pair, vocab, indices):
    """('a','b')페어를 ('ab')페어로 바꾸기 위한 [문장위치, ('ab') ('a','b'), 빈도수] 리스트 반환 """
    first, second = pair
    pair_str = ''.join(pair)
    pair_str = pair_str.replace('\\', '\\\\')
    changes = []
    pattern = re.compile(r'(?<!\S)' + re.escape(first + ' ' + second) + r'(?!\S)')
    iterator = indices[pair].items()
    for j, freq in iterator:
        if freq < 1:
            continue
        word, freq = vocab[j]
        new_word = ' '.join(word)
        new_word = pattern.sub(pair_str, new_word)
        new_word = tuple(new_word.split(' '))

        vocab[j] = (new_word, freq)
        changes.append((j, new_word, word, freq))
    return changes

def update_pair_statistics(pair, change, stats, indices):
    ''' pair 정보를 stats와 indices에서 삭제, 합쳐진 pair에 대한 모든 정보를 change에서 받아와 '''
    stats[pair] = 0
    indices[pair] = defaultdict(int)
    first, second = pair
    new_pair = first + second
    for j, word, old_word, freq in change:
        # find all instances of pair, and update frequency/indices around it
        i = 0
        while True:
            # find first symbol
            try:
                i = old_word.index(first, i)
            except ValueError:
                break
            # if first symbol is followed by second symbol, we've found an occurrence of pair (old_word[i:i+2])
            if i < len(old_word) - 1 and old_word[i + 1] == second:
                # assuming a symbol sequence "A B C", if "B C" is merged, reduce the frequency of "A B"
                if i:
                    prev = old_word[i - 1:i + 1]
                    stats[prev] -= freq
                    indices[prev][j] -= 1
                if i < len(old_word) - 2:
                    # assuming a symbol sequence "A B C B", if "B C" is merged, reduce the frequency of "C B".
                    # however, skip this if the sequence is A B C B C, because the frequency of "C B" will be reduced by the previous code block
                    if old_word[i + 2] != first or i >= len(old_word) - 3 or old_word[i + 3] != second:
                        nex = old_word[i + 1:i + 3]
                        stats[nex] -= freq
                        indices[nex][j] -= 1
                i += 2
            else:
                i += 1

        i = 0
        while True:
            try:
                # find new pair
                i = word.index(new_pair, i)
            except ValueError:
                break
            # assuming a symbol sequence "A BC D", if "B C" is merged, increase the frequency of "A BC"
            if i:
                prev = word[i - 1:i + 1]
                stats[prev] += freq
                indices[prev][j] += 1
            # assuming a symbol sequence "A BC B", if "B C" is merged, increase the frequency of "BC B"
            # however, if the sequence is A BC BC, skip this step because the count of "BC BC" will be incremented by the previous code block
            if i < len(word) - 1 and word[i + 1] != new_pair:
                nex = word[i:i + 2]
                stats[nex] += freq
                indices[nex][j] += 1
            i += 1

def save_file(file, args):
    os.mkdir(os.path.join(os.getcwd(), "preprocess/bpe_util_data"))
    rule_file = open(args.rule_outfile, "w", encoding="utf-8")
    for most_frequent in file:
        rule_file.write('{0} {1}\n'.format(*most_frequent))
    glossaries_file = open(args.glossaries_outfile, "w", encoding="utf-8")
    glossaries_file.close()

if __name__ == "__main__":
    args = set_hyperparameters()
    print("set hyperparameter")
    rules = learn_bpe(args)
    print("get rules")
    save_file(rules, args)
    print("save rules, glossaries file")
    print("Done.")