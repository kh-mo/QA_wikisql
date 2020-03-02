#-*- coding:utf-8 -*-

import os
import re
import copy
import json
import random
import argparse
from collections import Counter, defaultdict

def set_hyperparameters():
    parser = argparse.ArgumentParser()

    # file
    parser.add_argument("--rule_infile", default=os.path.join(os.getcwd(), "preprocess/bpe_util_data/rules.txt"))
    parser.add_argument("--glossaries_infile", default=os.path.join(os.getcwd(), "preprocess/bpe_util_data/glossaries.txt"))

    # option
    parser.add_argument("--separator", type=str, default="@@")
    parser.add_argument('--merges', type=int, default=3000, help="-1 은 모든 rule을 사용함을 의미")
    args = parser.parse_args()
    return args

def apply_bpe(args):
    if os.path.isdir(os.path.join(os.getcwd(), "preprocess/bpe")):
        pass
    else:
        os.mkdir(os.path.join(os.getcwd(), "preprocess/bpe"))
    types = ["train", "dev", "test"]
    for type in types:
        infile = open(os.path.join(os.getcwd(), "preprocess/stanford/" + type + "_s.jsonl"), "r", encoding="utf-8")
        outfile = open(os.path.join(os.getcwd(), "preprocess/bpe/" + type + "_sb_" + str(args.merges) + ".jsonl"), "w", encoding="utf-8")
        for line in infile:
            doc = json.loads(line)
            doc['nl_sb'] = bpe_process_line(doc['nl_s'], args)
            doc['col_sb'] = []
            for word in doc['col_s']:
                doc['col_sb'].append(bpe_process_line(word, args))
            outfile.write(json.dumps(doc, ensure_ascii=False))
            outfile.write("\n")

def bpe_process_line(line, args):
    ''' line의 앞뒤 공백을 제거한 후 space로 분할한 token을 segment_token 함수에 전달 '''
    segments = segment_tokens(line.strip('\r\n ').split(' '), args)
    return ' '.join(segments)

def segment_tokens(tokens, args):
    # rule 파일로부터 merge 갯수만큼 모든 규칙을 얻는다. merge가 -1이면 모든 규칙을 사용한다.
    rule = open(args.rule_infile, 'r', encoding='utf-8')
    bpe_rules = [tuple(item.strip('\r\n ').split(' ')) for (n, item) in enumerate(rule) if (args.merges == -1 or n < args.merges)]

    # { ('a','b'): 9999 ~ ('c','d'): 0 }
    # value가 낮을수록 규칙이 많이 등장했다.(learn bpe의 결과물 rule에서 상단에 있는 pair가 가장 많은 빈도로 trainset에서 발생한 조합임)
    args.bpe_rules = dict([(code, i) for (i, code) in reversed(list(enumerate(bpe_rules)))])

    # {'ab', ('a', 'b')), key는 합쳐진 것, value는 분할된 것
    args.bpe_rules_reverse = dict([(pair[0] + pair[1], pair) for pair, i in args.bpe_rules.items()])

    # subword로 쪼개지지 않을 단어 집합(회사에서 voca라고 부르는 항목)
    args.glossaries = open(args.glossaries_infile, 'r', encoding='utf-8').readlines()
    args.glossaries_regex = re.compile('^({})$'.format('|'.join(args.glossaries) + "|" + "|".join(["__" + word for word in args.glossaries]))) if args.glossaries else None

    output = []
    for token in tokens:
        # 띄어쓰기가 2개일 경우 공백이 tokens에 남는데 이것을 제거(eliminate double spaces)
        if not token:
            continue
        # 1. _isolate_glossaries() 먼저 수행
        # 2. 1에서 얻어진 segment를 받아 encode 수행
        # 3. 2에서 얻어진 encode 결과물 out을 list comprehension으로 저장
        new_word = [out for out in encode(token, args)]
        # new_word = [out for segment in _isolate_glossaries(token) for out in encode(segment, args)]
        # self.vocab
        # self.cache

        # token이 분할되어 new_word로 넘어오면 @@이 붙게되고, 분할되지 않으면 token이 그대로 output에 붙게된다.
        for item in new_word[:-1]:
            output.append(item + args.separator)
        output.append(new_word[-1])

    return output

# def _isolate_glossaries(word):
#     # 단어 중 glossaries안에 들어있는 subword는 분해되지 않고 유지된다
#     word_segments = [word]
#     for gloss in args.glossaries:
#         # 1. word_segments에서 segment 추출
#         # 2. segment와 gloss를 isolate_glossary에서 subword 매칭 여부 탐색
#         # 3. 2에서 얻어진 out_segments 결과물을 list comprehension으로 저장
#         word_segments = [out_segments for segment in word_segments
#                              for out_segments in isolate_glossary(segment, gloss)]
#     return word_segments
#
# def isolate_glossary(word, glossary):
#     """
#     Isolate a glossary present inside a word.
#     Returns a list of subwords. In which all 'glossary' glossaries are isolated
#     For example, if 'USA' is the glossary and '1934USABUSA' the word, the return value is:
#         ['1934', 'USA', 'B', 'USA']
#     """
#     # regex equivalent of (if word == glossary or glossary not in word)
#     if re.match('^'+glossary+'$', word) or not re.search(glossary, word):
#         return [word]
#     else:
#         segments = re.split(r'({})'.format(glossary), word)
#         segments, ending = segments[:-1], segments[-1]
#         segments = list(filter(None, segments)) # Remove empty strings in regex group.
#         return segments + [ending.strip('\r\n ')] if ending != '' else segments

def encode(token, args):
    """Encode word based on list of BPE merge operations, which are applied consecutively
    """

    # # dropout을 적용 안하고(and) 단어가 cache에 있는 경우
    # if not dropout and orig in cache:
    #     return cache[orig]

    # 단어가 regex에 매칭될 때 subword로 분해하지 않는다
    if args.glossaries_regex and args.glossaries_regex.match(token):
        # cache[token] = (token,)
        return (token,)

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
        pairs = [(args.bpe_rules[pair],i,pair) for (i,pair) in enumerate(zip(word, word[1:])) if pair in args.bpe_rules]

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
    apply_bpe(args)
    print("apply bpe")
    print("Done.")
