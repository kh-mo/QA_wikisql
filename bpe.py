#-*- coding:utf-8 -*-
'''
{ train, dev, test }set
dev를 이용해 OOV, unknown이 적은 규칙을 산출해야 함
최적의 규칙이 나오면 test에 적용
'''

import os
import re
import copy
import argparse
from collections import Counter, defaultdict

def get_vocabulary(file):
    vocab = Counter()
    for line in file:
        for word in line.strip("\r\n ").split(' '):
            if word:
                vocab[word] += 1
    return vocab

def get_pair_statistics(file):
    stats = defaultdict(int)
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

def learn_bpe(infile, outfile, min_frequency, num_symbols, total_symbols, verbose):
    # 단어의 마지막 토큰을 "</w>"로 지정한 버전입니다.
    outfile.write('#version: 0.2\n')

    # {단어 : count} 형태의 딕셔너리 반환합니다.
    vocab = get_vocabulary(infile)
    # 단어를 음절별로 나누기, 단어의 마지막 음절에는 </w>를 붙입니다.
    vocab = dict([(tuple(x[:-1])+(x[-1]+"</w>",) , y) for (x, y) in vocab.items()])
    # 단어를 count 기준으로 내림차순 정렬합니다.
    sorted_vocab = sorted(vocab.items(), key=lambda x:x[1], reverse=True)

    # stats : {[token1, token2]:count}, indices {[token1,token2]:{1번단어:count, 2번단어:count}}
    # stats는 특정 단어 페어가 등장한 총 수, indices는 해당 단어 페어가 등장한 위치
    stats, indices = get_pair_statistics(sorted_vocab)

    # 개별 유닛수를 구해 전체 단어 수에서 제외
    if total_symbols:
        uniq_char_internal = set()
        uniq_char_final = set()
        for word in vocab:
            for char in word[:-1]:
                uniq_char_internal.add(char)
            uniq_char_final.add(word[-1])
        num_symbols -= len(uniq_char_internal) + len(uniq_char_final)

    for i in range(num_symbols):
        # 가장 빈도가 높은 pair 획득
        if stats:
            most_frequent = max(stats, key=lambda x: stats[x])

        # 모든 pair들이 최소 빈도수 이하일 경우 종료
        if stats[most_frequent] < min_frequency:
            print('no pair has frequency >= {0}. Stopping\n'.format(min_frequency))
            break

        # 진행상황 보여주기
        if verbose:
            print('pair {0}: {1} {2} -> {1}{2} (frequency {3})\n'.format(
                i, most_frequent[0], most_frequent[1], stats[most_frequent]))

        outfile.write('{0} {1}\n'.format(*most_frequent))
        changes = replace_pair(most_frequent, sorted_vocab, indices)
        update_pair_statistics(most_frequent, changes, stats, indices)

    outfile.close()

class BPE(object):
    def __init__(self, rulefile, merges=-1, separator="@@"):

        # rule 파일로부터 merge 갯수만큼 모든 규칙을 얻는다. merge가 -1이면 모든 규칙을 사용한다.
        self.bpe_rules = [tuple(item.strip('\r\n ').split(' ')) for (n, item) in enumerate(rulefile) if (merges==-1 or n<merges)]

        self.bpe_rules = dict([(code, i) for (i, code) in reversed(list(enumerate(self.bpe_rules)))])
        self.bpe_rules_reverse = dict([(pair[0] + pair[1], pair) for pair, i in self.bpe_rules.items()])
        self.separator = separator
        self.cache = {}

    def process_line(self, line, dropout=0):
        out = ""
        leading_whitespace = len(line)-len(line.lstrip("\r\n "))
        if leading_whitespace:
            out += line[:leading_whitespace]
        out += self.segment(line, dropout)

        trailing_whitespace = len(line)-len(line.rstrip('\r\n '))
        if trailing_whitespace and trailing_whitespace != len(line):
            out += line[-trailing_whitespace:]

        return out

    def segment(self, sentence, dropout=0):
        segments = self.segment_token(sentence.strip('\r\n ').split(' '), dropout)
        return ' '.join(segments)

    def segment_tokens(self, tokens, dropout=0):
        output = []
        for word in tokens:
            if not word:
                continue
            new_word = [out for segment in self._isolate_glossaries(word) for out in encode(segment,
                                                                                            self.bpe_rules,
                                                                                            self.bpe_rules_reverse,
                                                                                            self.separator,
                                                                                            self.version,
                                                                                            self.cache,
                                                                                            dropout)]
            for item in new_word[:-1]:
                output.append(item + self.separator)
            output.append(new_word[-1])
        return output

def apply_bpe(instance, infile, outfile):
    for line in infile:
        outfile.write(instance.process_line(line))

if __name__ == "__main__":
    infile = open(os.path.join(os.getcwd(), "preprocess/train.txt"), "r", encoding="utf-8")
    outfile = open(os.path.join(os.getcwd(), "preprocess/codes.txt"), "w", encoding="utf-8")
    num_symbols = 10000 # 총 단어 수
    total_symbols = False # 총 단어 수에 개별 유닛(a,b,c 등) 포함 여부
    verbose = False # 진행경과를 보여줄 지 여부
    min_frequency = 2 # 종료 최소 빈도 수(등장한 유닛 pair들이 모두 이 기준보다 작다면 종료)
    learn_bpe(infile, outfile, min_frequency, num_symbols, total_symbols, verbose)

    infile = open(os.path.join(os.getcwd(), "preprocess/train.txt"), "r", encoding="utf-8")
    rulefile = open(os.path.join(os.getcwd(), "preprocess/codes.txt"), "r", encoding="utf-8")
    resultfile = open(os.path.join(os.getcwd(), "preprocess/train_bpe.txt"), "w", encoding="utf-8")
    bpe = BPE(rulefile, -1, "@@")
    bpe.bpe_rules
    apply_bpe(bpe, infile, resultfile)




from __future__ import unicode_literals, division

import sys
import os
import inspect
import codecs
import io
import argparse
import re
import warnings
import random


# hack for python2/3 compatibility
from io import open
argparse.open = open

class BPE(object):

    def __init__(self, codes, merges=-1, separator='@@', vocab=None, glossaries=None):

        codes.seek(0)
        offset=1

        # check version information
        firstline = codes.readline()
        if firstline.startswith('#version:'):
            self.version = tuple([int(x) for x in re.sub(r'(\.0+)*$','', firstline.split()[-1]).split(".")])
            offset += 1
        else:
            self.version = (0, 1)
            codes.seek(0)

        self.bpe_codes = [tuple(item.strip('\r\n ').split(' ')) for (n, item) in enumerate(codes) if (n < merges or merges == -1)]

        for i, item in enumerate(self.bpe_codes):
            if len(item) != 2:
                sys.stderr.write('Error: invalid line {0} in BPE codes file: {1}\n'.format(i+offset, ' '.join(item)))
                sys.stderr.write('The line should exist of exactly two subword units, separated by whitespace\n')
                sys.exit(1)

        # some hacking to deal with duplicates (only consider first instance)
        self.bpe_codes = dict([(code,i) for (i,code) in reversed(list(enumerate(self.bpe_codes)))])

        self.bpe_codes_reverse = dict([(pair[0] + pair[1], pair) for pair,i in self.bpe_codes.items()])

        self.separator = separator

        self.vocab = vocab

        self.glossaries = glossaries if glossaries else []

        self.glossaries_regex = re.compile('^({})$'.format('|'.join(glossaries))) if glossaries else None

        self.cache = {}

    def process_line(self, line, dropout=0):
        """segment line, dealing with leading and trailing whitespace"""

        out = ""

        leading_whitespace = len(line)-len(line.lstrip('\r\n '))
        if leading_whitespace:
            out += line[:leading_whitespace]

        out += self.segment(line, dropout)

        trailing_whitespace = len(line)-len(line.rstrip('\r\n '))
        if trailing_whitespace and trailing_whitespace != len(line):
            out += line[-trailing_whitespace:]

        return out

    def segment(self, sentence, dropout=0):
        """segment single sentence (whitespace-tokenized string) with BPE encoding"""
        segments = self.segment_tokens(sentence.strip('\r\n ').split(' '), dropout)
        return ' '.join(segments)

    def segment_tokens(self, tokens, dropout=0):
        """segment a sequence of tokens with BPE encoding"""
        output = []
        for word in tokens:
            # eliminate double spaces
            if not word:
                continue
            new_word = [out for segment in self._isolate_glossaries(word)
                        for out in encode(segment,
                                          self.bpe_codes,
                                          self.bpe_codes_reverse,
                                          self.vocab,
                                          self.separator,
                                          self.version,
                                          self.cache,
                                          self.glossaries_regex,
                                          dropout)]

            for item in new_word[:-1]:
                output.append(item + self.separator)
            output.append(new_word[-1])

        return output

    def _isolate_glossaries(self, word):
        word_segments = [word]
        for gloss in self.glossaries:
            word_segments = [out_segments for segment in word_segments
                                 for out_segments in isolate_glossary(segment, gloss)]
        return word_segments

def create_parser(subparsers=None):

    if subparsers:
        parser = subparsers.add_parser('apply-bpe',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description="learn BPE-based word segmentation")
    else:
        parser = argparse.ArgumentParser(
            formatter_class=argparse.RawDescriptionHelpFormatter,
            description="learn BPE-based word segmentation")

    parser.add_argument(
        '--input', '-i', type=argparse.FileType('r'), default=sys.stdin,
        metavar='PATH',
        help="Input file (default: standard input).")
    parser.add_argument(
        '--codes', '-c', type=argparse.FileType('r'), metavar='PATH',
        required=True,
        help="File with BPE codes (created by learn_bpe.py).")
    parser.add_argument(
        '--merges', '-m', type=int, default=-1,
        metavar='INT',
        help="Use this many BPE operations (<= number of learned symbols)"+
             "default: Apply all the learned merge operations")
    parser.add_argument(
        '--output', '-o', type=argparse.FileType('w'), default=sys.stdout,
        metavar='PATH',
        help="Output file (default: standard output)")
    parser.add_argument(
        '--separator', '-s', type=str, default='@@', metavar='STR',
        help="Separator between non-final subword units (default: '%(default)s'))")
    parser.add_argument(
        '--vocabulary', type=argparse.FileType('r'), default=None,
        metavar="PATH",
        help="Vocabulary file (built with get_vocab.py). If provided, this script reverts any merge operations that produce an OOV.")
    parser.add_argument(
        '--vocabulary-threshold', type=int, default=None,
        metavar="INT",
        help="Vocabulary threshold. If vocabulary is provided, any word with frequency < threshold will be treated as OOV")
    parser.add_argument(
        '--dropout', type=float, default=0,
        metavar="P",
        help="Dropout BPE merge operations with probability P (Provilkov et al., 2019). Use this on training data only.")
    parser.add_argument(
        '--glossaries', type=str, nargs='+', default=None,
        metavar="STR",
        help="Glossaries. Words matching any of the words/regex provided in glossaries will not be affected "+
             "by the BPE (i.e. they will neither be broken into subwords, nor concatenated with other subwords. "+
             "Can be provided as a list of words/regex after the --glossaries argument. Enclose each regex in quotes.")

    return parser

def encode(orig, bpe_codes, bpe_codes_reverse, vocab, separator, version, cache, glossaries_regex=None, dropout=0):
    """Encode word based on list of BPE merge operations, which are applied consecutively
    """

    if not dropout and orig in cache:
        return cache[orig]

    if glossaries_regex and glossaries_regex.match(orig):
        cache[orig] = (orig,)
        return (orig,)

    if len(orig) == 1:
        return orig

    if version == (0, 1):
        word = list(orig) + ['</w>']
    elif version == (0, 2): # more consistent handling of word-final segments
        word = list(orig[:-1]) + [orig[-1] + '</w>']
    else:
        raise NotImplementedError

    while len(word) > 1:

        # get list of symbol pairs; optionally apply dropout
        pairs = [(bpe_codes[pair],i,pair) for (i,pair) in enumerate(zip(word, word[1:])) if (not dropout or random.random() > dropout) and pair in bpe_codes]

        if not pairs:
            break

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
    if vocab:
        word = check_vocab_and_split(word, bpe_codes_reverse, vocab, separator)

    cache[orig] = word
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

def read_vocabulary(vocab_file, threshold):
    """read vocabulary file produced by get_vocab.py, and filter according to frequency threshold.
    """

    vocabulary = set()

    for line in vocab_file:
        word, freq = line.strip('\r\n ').split(' ')
        freq = int(freq)
        if threshold == None or freq >= threshold:
            vocabulary.add(word)

    return vocabulary

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

if __name__ == '__main__':

    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    newdir = os.path.join(currentdir, 'subword_nmt')
    if os.path.isdir(newdir):
        warnings.simplefilter('default')
        warnings.warn(
            "this script's location has moved to {0}. This symbolic link will be removed in a future version. Please point to the new location, or install the package and use the command 'subword-nmt'".format(newdir),
            DeprecationWarning
        )

    # python 2/3 compatibility
    if sys.version_info < (3, 0):
        sys.stderr = codecs.getwriter('UTF-8')(sys.stderr)
        sys.stdout = codecs.getwriter('UTF-8')(sys.stdout)
        sys.stdin = codecs.getreader('UTF-8')(sys.stdin)
    else:
        sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', write_through=True, line_buffering=True)

    parser = create_parser()
    args = parser.parse_args()

    # read/write files as UTF-8
    args.codes = codecs.open(args.codes.name, encoding='utf-8')
    if args.input.name != '<stdin>':
        args.input = codecs.open(args.input.name, encoding='utf-8')
    if args.output.name != '<stdout>':
        args.output = codecs.open(args.output.name, 'w', encoding='utf-8')
    if args.vocabulary:
        args.vocabulary = codecs.open(args.vocabulary.name, encoding='utf-8')

    if args.vocabulary:
        vocabulary = read_vocabulary(args.vocabulary, args.vocabulary_threshold)
    else:
        vocabulary = None

    if sys.version_info < (3, 0):
        args.separator = args.separator.decode('UTF-8')
        if args.glossaries:
            args.glossaries = [g.decode('UTF-8') for g in args.glossaries]

    bpe = BPE(args.codes, args.merges, args.separator, vocabulary, args.glossaries)

    for line in args.input:
        args.output.write(bpe.process_line(line, args.dropout))