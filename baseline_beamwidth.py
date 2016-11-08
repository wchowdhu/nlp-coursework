#!/usr/bin/env python
import optparse
import sys
import models
from collections import namedtuple
from math import log10


# Finding the appropriate beam stack with
# value sorted according to their probabilities

def find_beam(stack):
    beam = sorted(stack.itervalues(), key=lambda h: -h.logprob)
    max = beam[0].logprob
    # beam_width = (max + min)/2.0
    # beam_filtered = [hypothesis for hypothesis in beam if hypothesis.logprob >= beam_width]
    # beam_width = log10(0.00000001) = -8
    beam_width = log10(0.00000001)
    beam_filtered = [hypothesis     for hypothesis in beam     if hypothesis.logprob >= (max + beam_width)]
    # beam_filtered = [hypothesis     for hypothesis in beam      if abs(max - hypothesis.logprob) <= abs(beam_width)]
    # returns list of sorted hypothesis/states according to logprob and filtered according to beam width
    return beam_filtered


# The following code implements a beam decoding
# algorithm with reordering (one that permutes the target
# phrases). Hence all hypotheses in stacks[i] represent
# translations of *any* i words.

def beam_stack_decode(french, tm, lm, opts):
    sys.stderr.write("Decoding %s...\n" % (opts.input,))

    # for each index,tuple of words (e.g each french input sentence) in french list
    for s_no, f in enumerate(french):
        sys.stderr.write("French Sentence --> %s...\n" % str(int(s_no) + 1))
        # Tweaking params based on the len of french sen
        opt_eta = opts.eta
        opt_s = opts.s
        opt_distort = opts.distort
        opt_k = opts.k
        if len(f) < 9:
            opt_eta = 0.5
            opt_s = 200
            opt_distort = 3
        if len(f) > 15:
            opt_eta = 0.9
            opt_s = 500
            opt_distort = 3
            opt_k = 20

        bit_vec = [0] * len(f)

        # Initialising the hypothesis
        # logprob = previous state logprob + phrase logprob, lm_state = last 2 words of the phrase,
        # last_ind = index of the last word in last phrase in previous state, predecessor = previous state
        hypothesis = namedtuple("hypothesis", "logprob, lm_state, bit_vec, last_ind, predecessor, phrase")
        initial_hypothesis = hypothesis(0.0, lm.begin(), bit_vec, 0, None, None)

        # Creating the stacks
        stacks = [{} for _ in f] + [{}]
        stacks[0][lm.begin()] = initial_hypothesis

        # Iterating over all stacks except the one where all words
        # are translated
        for i, stack in enumerate(stacks[:-1]):

            # Find the sorted and pruned stack
            # apply beam limit
            beam = find_beam(stack)

            # Iterating on Pruned Hypotheses - Histogram Pruning
            # beam[:opt_s] = take top opt_s states per stack
            # instead apply beam width limit here e.g limit on number of states to consider per stack
            for h in beam:

                # Iterating over all phrase possibilities
                probable_phrases = []
                prob_dist_phrases = []

                # ph_range consists of all valid phrases that can follow the h/hypothesis/state
                # x = starting french/source index of the english phrase
                # y = ending french/source index of the english phrase
                ph_range = namedtuple("ph_range", "x, y")
                for x in xrange(0, len(f)):
                    for y in xrange(x + 1, len(f) + 1):
                        if 1 in h.bit_vec[x:y]:
                            continue
                        # checking if consecutive phrases are close to each other using distortion limit of 9
                        if abs(h.last_ind + 1 - x) > 9:
                            prob_dist_phrases.append(ph_range(x, y))
                        else:
                            probable_phrases.append(ph_range(x, y))

                if len(probable_phrases) == 0:
                    probable_phrases = prob_dist_phrases[:]

                for phrase_range in probable_phrases:

                    f_phrase = f[phrase_range.x:phrase_range.y]
                    if f_phrase in tm:

                        # Deep copying the bit vector
                        new_bit_vec = h.bit_vec[:]
                        for bt in xrange(phrase_range.x, phrase_range.y):
                            new_bit_vec[bt] = 1

                        for phrase in tm[f_phrase][:opt_k]:

                            # Adding the phrase translation probability
                            logprob = h.logprob + phrase.logprob
                            lm_state = h.lm_state

                            # Computing the language probability for english phrase
                            for word in phrase.english.split():
                                (lm_state, word_logprob) = lm.score(lm_state, word)
                                logprob += word_logprob
                            if 0 not in new_bit_vec:
                                logprob += lm.end(lm_state)
                            logprob += log10(opt_eta) * abs(h.last_ind + 1 - phrase_range.x)

                            # Check for the correct stack number
                            # new_bit_vec.count(1) returns how many time 1 occurs in new_bit vector (e.g number of translated words), that number will be the stack number
                            if i + phrase_range.y - phrase_range.x != new_bit_vec.count(1):
                                sys.stderr.write("Stack Error")
                            # Create the new hypothesis
                            new_hypothesis = hypothesis(logprob, lm_state, new_bit_vec, phrase_range.y - 1, h, phrase)

                            # Recombination in the stack
                            # Add method in algorithm
                            if lm_state not in stacks[i + phrase_range.y - phrase_range.x] or \
                                            stacks[i + phrase_range.y - phrase_range.x][
                                                lm_state].logprob < logprob:  # second case is recombination
                                stacks[i + phrase_range.y - phrase_range.x][lm_state] = new_hypothesis

        winner = max(stacks[-1].itervalues(), key=lambda h: h.logprob)

        print extract_english(winner)

        if opts.verbose:
            tm_logprob = extract_tm_logprob(winner)
            sys.stderr.write("LM = %f, TM = %f, Total = %f\n" %
                             (winner.logprob - tm_logprob, tm_logprob, winner.logprob))

    return


def extract_english(h):
    return "" if h.predecessor is None else "%s%s " % (extract_english(h.predecessor), h.phrase.english)


def extract_tm_logprob(h):
    return 0.0 if h.predecessor is None else h.phrase.logprob + extract_tm_logprob(h.predecessor)


if __name__ == "__main__":
    optparser = optparse.OptionParser()
    optparser.add_option("-i", "--input", dest="input", default="data/input",
                         help="File containing sentences to translate (default=data/input)")
    optparser.add_option("-t", "--translation-model", dest="tm", default="data/tm",
                         help="File containing translation model (default=data/tm)")
    optparser.add_option("-l", "--language-model", dest="lm", default="data/lm",
                         help="File containing ARPA-format language model (default=data/lm)")
    optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int",
                         help="Number of sentences to decode (default=no limit)")
    optparser.add_option("-k", "--translations-per-phrase", dest="k", default=1, type="int",
                         help="Limit on number of translations to consider per phrase (default=1)")
    optparser.add_option("-s", "--stack-size", dest="s", default=1, type="int", help="Maximum stack size (default=1)")
    optparser.add_option("-e", "--eta", dest="eta", default=0.6, type="float",
                         help="Eta Value for distortion model (default=0.6)")
    optparser.add_option("-d", "--distort", dest="distort", default=4, type="int",
                         help="Maximum distortion length (default=4)")
    optparser.add_option("-a", "--alpha", dest="alpha", default=0.0001, type="float",
                         help="Alpha value for threshold pruning (default=0.0001)")
    optparser.add_option("-v", "--verbose", dest="verbose", action="store_true", default=False,
                         help="Verbose mode (default=off)")
    opts = optparser.parse_args()[0]

    tm = models.TM(opts.tm, opts.k)
    lm = models.LM(opts.lm)

    # Getting the French sentences
    french = [tuple(line.strip().split()) for line in open(opts.input).readlines()[:opts.num_sents]]

    # tm should translate unknown words as-is with probability 1
    for word in set(sum(french, ())):
        if (word,) not in tm:
            tm[(word,)] = [models.phrase(word, 0.0)]

    beam_stack_decode(french, tm, lm, opts)
