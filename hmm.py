from enum import Enum
from itertools import *

import numpy as np
from itertools import islice

_training_data = "/Users/junix/nlp/icwb2-data/training/msr_training.utf8"


class State(Enum):
    B = 0
    M = 1
    E = 2
    S = 3

    @classmethod
    def states(cls):
        return tuple(State(i) for i in State.int_states())

    @classmethod
    def int_states(cls):
        return tuple(range(4))

    @classmethod
    def state_count(cls):
        return len(cls.states())


class CharSet:
    def __init__(self, charseq):
        self.ch2ix = {}
        self.ix2ch = {}
        for c in charseq:
            if c not in self.ch2ix:
                index = len(self.ch2ix)
                self.ch2ix[c] = index
                self.ix2ch[index] = c

    def __getitem__(self, item):
        if isinstance(item, str):
            return self.ch2ix[item]
        elif isinstance(item, int):
            return self.ix2ch[item]
        else:
            assert "invalid item:{}".format(item)

    def __len__(self):
        return len(self.ch2ix)


def load_training_set():
    def do_label(ts):
        for token in ts:
            if not token:
                continue
            h, *left = token
            if not left:
                yield h, State.S
            else:
                yield h, State.B
                for c in left[:-1]:
                    yield c, State.M
                yield left[-1], State.E

    with open(_training_data, 'r', encoding='utf-8') as f:
        for line in f:
            tokens = line.strip('\n').split()
            yield tuple(do_label(tokens))


class HMM:
    def __init__(self, init_states, trans_prob_matrix, emit_prob_matrix, charset):
        self.inits = init_states
        self.trans = trans_prob_matrix
        self.emits = emit_prob_matrix
        self.charset = charset

    def init_state(self, s):
        if isinstance(s, int):
            return self.inits[s]
        return self.inits[s.value]

    def viterbi(self, text):
        int_text = [self.charset[c] for c in text]
        step_state_probs = np.zeros((len(int_text), State.state_count()))
        path = {}

        # Initialize base cases (t == 0)
        for s in State.int_states():
            step_state_probs[0][s] = self.inits[s] * self.emits[s][int_text[0]]
            path[s] = [s]

        # Run Viterbi for t > 0
        for t, int_char in islice(enumerate(int_text), 1, None):
            new_path = {}

            for s in State.int_states():
                prob, prev_s = max([
                    (step_state_probs[t - 1][s0] * self.trans[s0][s] * self.emits[s][int_char], s0)
                    for s0 in State.int_states()])
                step_state_probs[t][s] = prob
                new_path[s] = path[prev_s] + [s]

            # Don't need to remember the old paths
            path = new_path

        prob, prev_s = max([(step_state_probs[len(text) - 1][s], s) for s in State.int_states()])
        return path[prev_s]

    def cut(self, text):
        acc = []
        for c, s in zip(text, self.viterbi(text)):
            acc.append(c)
            if s in (State.S.value, State.E.value):
                yield ''.join(acc)
                acc = []


def build_charset(train_set):
    def _char_seq():
        for cs in train_set:
            for c, _ in cs:
                yield c

    return CharSet(_char_seq())


def _calc_init_freq(training_set):
    inits = np.zeros(State.state_count(), dtype=np.float)
    for sentence in training_set:
        if not sentence:
            continue
        (_, state), *_ = sentence
        inits[state.value] += 1
    return inits / np.sum(inits)


def _calc_trans_freq(training_set):
    def _trans_seq():
        for sentence in training_set:
            states = [s for _, s in sentence]
            yield from zip(states, states[1:])

    cnt = State.state_count()
    freqs = np.zeros((cnt, cnt))
    for beg, end in _trans_seq():
        freqs[beg.value][end.value] += 1
    total_cnt = np.sum(freqs)
    return freqs / total_cnt


def _calc_emit_freq(training_set, charset):
    def _emit_seq():
        for sentence in training_set:
            yield from sentence

    cnt = State.state_count()
    freqs = np.zeros((cnt, len(charset)))
    for ch, state in _emit_seq():
        freqs[state.value][charset[ch]] += 1
    total_cnt = np.sum(freqs)
    return freqs / total_cnt


def build_hmm():
    train_set = tuple(load_training_set())
    charset = build_charset(train_set)
    init_freq = _calc_init_freq(train_set)
    trans_freq = _calc_trans_freq(train_set)
    emit_freq = _calc_emit_freq(train_set, charset)
    return HMM(init_states=init_freq, trans_prob_matrix=trans_freq, emit_prob_matrix=emit_freq, charset=charset)


if __name__ == '__main__':
    hmm = build_hmm()
    sentence = '然而那篇文章显然有很多不足的地方，比如介绍不够清晰，也不够完整，还没有实现，在这里我们重提这个模型，将相关内容补充完成。'
    print('/'.join(hmm.cut(sentence)))
