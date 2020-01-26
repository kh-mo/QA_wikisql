#-*- coding:utf-8 -*-
'''
{ train, dev, test }set
dev를 이용해 OOV, unknown이 적은 규칙을 산출해야 함
최적의 규칙이 나오면 test에 적용

oov, voca, glossary 추가 필요
'''

import os
import re
import copy
import random
import argparse
from collections import Counter, defaultdict

def set_hyperparameters():
    parser = argparse.ArgumentParser()

    # file
    parser.add_argument("--train_infile", default=os.path.join(os.getcwd(), "preprocess/train_token_basic.txt"))
    parser.add_argument("--voca", default=os.path.join(os.getcwd(), "preprocess/voca.txt"))
    parser.add_argument("--rule_file", default=os.path.join(os.getcwd(), "preprocess/rules.txt"))

    # option
    parser.add_argument("--get_rules", type=str, default="False")
    parser.add_argument("--include_character_unit", action="store_true")
    parser.add_argument("--num_symbols", type=int, default=10000000)
    parser.add_argument("--min_frequency", type=int, default=2)
    parser.add_argument("--verbose", type=str, default="false")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--glossaries", type=str, nargs='+', default=None)
    parser.add_argument("--separator", type=str, default="@@")
    parser.add_argument('--merges', type=int, default=3000, help="-1 은 모든 rule을 사용함을 의미")
    args = parser.parse_args()
    return args

def learn_bpe(args):
    infile = open(args.train_infile, "r", encoding="utf-8")
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
            print('pair {0}: {1} {2} -> {1}{2} (frequency {3})\n'.format(
                i, most_frequent[0], most_frequent[1], stats[most_frequent]))

        output.append(most_frequent)
        changes = replace_pair(most_frequent, sorted_vocab, indices)
        update_pair_statistics(most_frequent, changes, stats, indices)

    infile.close()

    return output

def get_vocabulary(file):
    vocab = Counter()
    for line in file:
        for word in line.strip("\r\n ").split(' '):
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

def save_rule_file(file, args):
    outfile = open(args.rule_file, "w", encoding="utf-8")
    for most_frequent in file:
        outfile.write('{0} {1}\n'.format(*most_frequent))

def apply_bpe(args):
    types = ["train", "dev", "test"]
    for type in types:
        infile = open(os.path.join(os.getcwd(), "preprocess/" + type + "_token_basic.txt"), "r", encoding="utf-8")
        outfile = open(os.path.join(os.getcwd(), "preprocess/" + type + "_bpe_" + str(args.merges) + ".txt"), "w", encoding="utf-8")
        for line in infile:
            outfile.write(bpe_process_line(line, args))

def bpe_process_line(line, args):
    '''
    out = "왼쪽 공백" + self.segment + "오른쪽 공백"
    오른쪽 공백의 if문이 2가지 조건을 가지는 이유는
    1. segment로 나눌 line이 있는 경우는 공백만 계산하면 되고
    2. line이 공백인 경우 왼쪽공백을 구하는 쪽에서 모든 공백을 이미 계산하기 때문이다.
    '''
    out = ""

    leading_whitespace = len(line) - len(line.lstrip("\r\n "))
    if leading_whitespace:
        out += line[:leading_whitespace]

    out += segment(line, args)

    trailing_whitespace = len(line) - len(line.rstrip('\r\n '))
    if trailing_whitespace and trailing_whitespace != len(line):
        out += line[-trailing_whitespace:]

    return out

def segment(line, args):
    '''
    line의 앞뒤 공백을 제거한 후 space로 분할한 token을 segment_token 함수에 전달
    '''
    segments = segment_tokens(line.strip('\r\n ').split(' '), args)
    return ' '.join(segments)

def segment_tokens(tokens, args):
    # rule 파일로부터 merge 갯수만큼 모든 규칙을 얻는다. merge가 -1이면 모든 규칙을 사용한다.
    rule = open(args.rule_file, 'r', encoding='utf-8')
    bpe_rules = [tuple(item.strip('\r\n ').split(' ')) for (n, item) in enumerate(rule) if (args.merges == -1 or n < args.merges)]

    # { ('a','b'): 9999 ~ ('c','d'): 0 }
    # value가 낮을수록 규칙이 많이 등장했다.(learn bpe의 결과물 rule에서 상단에 있는 pair가 가장 많은 빈도로 trainset에서 발생한 조합임)
    args.bpe_rules = dict([(code, i) for (i, code) in reversed(list(enumerate(bpe_rules)))])

    # {'ab', ('a', 'b')), key는 합쳐진 것, value는 분할된 것
    args.bpe_rules_reverse = dict([(pair[0] + pair[1], pair) for pair, i in args.bpe_rules.items()])

    # glossaries 추가 고려(option)
    args.glossaries = []

    output = []
    for token in tokens:
        # 띄어쓰기가 2개일 경우 공백이 tokens에 남는데 이것을 제거(eliminate double spaces)
        if not token:
            continue
        # 1. _isolate_glossaries() 먼저 수행
        # 2. 1에서 얻어진 segment를 받아 encode 수행
        # 3. 2에서 얻어진 encode 결과물 out을 list comprehension으로 저장
        new_word = [out for segment in _isolate_glossaries(token) for out in encode(segment, args)]
        # self.vocab
        # self.cache

        # token이 분할되어 new_word로 넘어오면 @@이 붙게되고, 분할되지 않으면 token이 그대로 output에 붙게된다.
        for item in new_word[:-1]:
            output.append(item + args.separator)
        output.append(new_word[-1])

    return output

def _isolate_glossaries(word):
    # 단어 중 glossaries안에 들어있는 subword는 분해되지 않고 유지된다
    word_segments = [word]
    for gloss in args.glossaries:
        # 1. word_segments에서 segment 추출
        # 2. segment와 gloss를 isolate_glossary에서 subword 매칭 여부 탐색
        # 3. 2에서 얻어진 out_segments 결과물을 list comprehension으로 저장
        word_segments = [out_segments for segment in word_segments
                             for out_segments in isolate_glossary(segment, gloss)]
    return word_segments

def isolate_glossary(word, glossary):
    """
    Isolate a glossary present inside a word.
    Returns a list of subwords. In which all 'glossary' glossaries are isolated
    For example, if 'USA' is the glossary and '1934USABUSA' the word, the return value is:
        ['1934', 'USA', 'B', 'USA']
    """
    # regex equivalent of (if word == glossary or glossary not in word)
    if re.match('^'+glossary+'$', word) or not re.search(glossary, word):
        return [word]
    else:
        segments = re.split(r'({})'.format(glossary), word)
        segments, ending = segments[:-1], segments[-1]
        segments = list(filter(None, segments)) # Remove empty strings in regex group.
        return segments + [ending.strip('\r\n ')] if ending != '' else segments

def encode(token, args):
    """Encode word based on list of BPE merge operations, which are applied consecutively
    """

    # # dropout을 적용 안하고(and) 단어가 cache에 있는 경우
    # if not dropout and orig in cache:
    #     return cache[orig]

    # # regex가 존재하고(and) 단어가 regex에 매칭될 때
    # if glossaries_regex and glossaries_regex.match(orig):
    #     cache[orig] = (orig,)
    #     return (orig,)

    # 단어가 단일 character일 때
    if len(token) == 1:
        return token

    word = list(token[:-1]) + [token[-1] + '</w>']

    # 위의 조건들을 다 pass 했을 경우
    while len(word) > 1:
        '''
        break 조건
        1. 단어가 다 합쳐져서 단일 단어가 되는 경우
        2. 합칠 pairs가 없는 경우
        '''
        # dropout을 고려하고(and) pair가 bpe_codes에 있는 (bpe_codes[pair], i, pair) list comprehension
        # get list of symbol pairs; optionally apply dropout
        pairs = [(args.bpe_rules[pair],i,pair) for (i,pair) in enumerate(zip(word, word[1:])) if (not args.dropout or random.random() > args.dropout) and pair in args.bpe_rules]

        # pairs가 빈 리스트면 break
        if not pairs:
            break

        # 우선순위가 제일 높은 pair, 첫 iter에선 대부분 character 조합이 나온다 => ('a', 'b')
        #get first merge operation in list of BPE codes
        bigram = min(pairs)[2]

        # find start position of all pairs that we want to merge
        positions = [i for (rank,i,pair) in pairs if pair == bigram]

        i = 0
        new_word = []
        bigram = ''.join(bigram)
        for j in positions:
            # merges are invalid if they start before current position. This can happen if there are overlapping pairs: (x x x -> xx x)
            if j < i:
                continue
            new_word.extend(word[i:j]) # all symbols before merged pair
            new_word.append(bigram) # merged pair
            i = j+2 # continue after merged pair
        new_word.extend(word[i:]) # add all symbols until end of word
        word = new_word

    # don't print end-of-word symbols
    if word[-1] == '</w>':
        word = word[:-1]
    elif word[-1].endswith('</w>'):
        word[-1] = word[-1][:-4]

    word = tuple(word)
    # vocab = read_vocabulary(open(args.voca, 'r', encoding='utf-8'))
    # if vocab:
    #     word = check_vocab_and_split(word, args.bpe_rules_reverse, vocab, args.separator)

    # cache[orig] = word
    return word

def recursive_split(segment, bpe_codes, vocab, separator, final=False):
    """Recursively split segment into smaller units (by reversing BPE merges)
    until all units are either in-vocabulary, or cannot be split futher."""

    try:
        if final:
            left, right = bpe_codes[segment + '</w>']
            right = right[:-4]
        else:
            left, right = bpe_codes[segment]
    except:
        #sys.stderr.write('cannot split {0} further.\n'.format(segment))
        yield segment
        return

    if left + separator in vocab:
        yield left
    else:
        for item in recursive_split(left, bpe_codes, vocab, separator, False):
            yield item

    if (final and right in vocab) or (not final and right + separator in vocab):
        yield right
    else:
        for item in recursive_split(right, bpe_codes, vocab, separator, final):
            yield item

def check_vocab_and_split(orig, bpe_codes, vocab, separator):
    """Check for each segment in word if it is in-vocabulary,
    and segment OOV segments into smaller units by reversing the BPE merge operations"""

    out = []

    for segment in orig[:-1]:
        if segment + separator in vocab:
            out.append(segment)
        else:
            #sys.stderr.write('OOV: {0}\n'.format(segment))
            for item in recursive_split(segment, bpe_codes, vocab, separator, False):
                out.append(item)

    segment = orig[-1]
    if segment in vocab:
        out.append(segment)
    else:
        #sys.stderr.write('OOV: {0}\n'.format(segment))
        for item in recursive_split(segment, bpe_codes, vocab, separator, True):
            out.append(item)

    return out

def read_vocabulary(vocab_file):
    """read vocabulary file produced by get_vocab.py, and filter according to frequency threshold.
    """

    vocabulary = set()

    for line in vocab_file:
        word, freq = line.strip('\r\n ').split(' ')
        freq = int(freq)
        vocabulary.add(word)

    return vocabulary

if __name__ == "__main__":
    args = set_hyperparameters()
    print("set hyperparameter")
    if args.get_rules == "True":
        rules = learn_bpe(args)
        print("get rules")
        save_rule_file(rules, args)
        print("save rules")
    apply_bpe(args)
    print("apply bpe")
    print("Done.")
