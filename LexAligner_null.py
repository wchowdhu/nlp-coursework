#!/usr/bin/env python
from __future__ import division
import optparse
import sys
import os
import logging
import copy
from collections import defaultdict


def Lexical_Aligner(bitext, numepochs):
    # f_count is a dict with key= french word and value=# of occcurences of the word
    f_count = defaultdict(int)
    # e_count is a dict with key=english word and value=# of occcurences of the word
    e_count = defaultdict(int)
    # fe_count is a dict with key=tuple(f_word,e_word) and value=# of occcurences of both the words in parallel corpus
    fe_count = defaultdict(int)

    # for each f-e word pair lists in bitext
    for (n, (f, e)) in enumerate(bitext):
        f_ = set(f)
        e_ = set(e)
        e_.add(None)
        # for each distinct word in french sentence, f
        for f_i in f_:
            # store the # of times the word f_i occurs in the parallel corpus
            f_count[f_i] += 1
            # for each distinct word in english sentence, e
            for e_j in e_:
                # set the initial # of times both the words occur in the parallel corpus to 0
                fe_count[(f_i, e_j)] = 0
        # for each distinct word in english sentence, e
        for e_j in e_:
            # set the initial # of times the word e_i occurs in the parallel corpus to 0
            e_count[e_j] = 0
        if n % 500 == 0:
            sys.stderr.write(".")
    # ==========================================
    #                   Training
    # ==========================================
    t = fe_count
    # Initializing values of t uniformly means that every French word is equally likely for every English word
    t = dict.fromkeys(t, float(1/len(f_count)))
    # for each iteration
    for i in range(numepochs):
        # creates a new dictionary with keys from e_count and values set to 0
        e_count = dict.fromkeys(e_count, 0)
        # creates a new dictionary with keys from fe_count and values set to 0
        fe_count = dict.fromkeys(fe_count, 0)
        for (f, e) in bitext:
            e_ = copy.copy(e)
            e_.insert(0,None)
            for f_i in f:
                Z = 0
                for (j, e_j) in enumerate(e_):
                    Z += t[(f_i, e_j)]
                for (j, e_j) in enumerate(e_):
                    c = float(t[(f_i, e_j)]/Z)
                    fe_count[(f_i, e_j)] += c
                    e_count[(e_j)] += c
        for (f_i, e_j) in fe_count:
            t[(f_i, e_j)] = float(fe_count[(f_i, e_j)]/e_count[(e_j)])
    # ==========================================
    #                   Decoding
    # ==========================================
    for (f, e) in bitext:
        for (i, f_i) in enumerate(f):
            bestp = t[(f_i, None)]
            bestj = 0
            for (j, e_j) in enumerate(e):
                if t[(f_i, e_j)] > bestp:
                    bestp = t[(f_i, e_j)]
                    bestj = j
            sys.stdout.write("%i-%i " % (i, bestj))
        sys.stdout.write("\n")


if __name__ == '__main__':

    optparser = optparse.OptionParser()
    optparser.add_option("-d", "--datadir", dest="datadir", default="data", help="data directory (default=data)")
    optparser.add_option("-p", "--prefix", dest="fileprefix", default="hansards", help="prefix of parallel data files (default=hansards)")
    optparser.add_option("-e", "--english", dest="english", default="en", help="suffix of English filename (default=en)")
    optparser.add_option("-f", "--french", dest="french", default="fr", help="suffix of French (source language) filename (default=fr)")
    optparser.add_option("-l", "--logfile", dest="logfile", default=None, help="filename for logging output")
    optparser.add_option("-t", "--threshold", dest="threshold", default=0.5, type="float", help="threshold for alignment (default=0.5)")
    optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to use for training and alignment")
    optparser.add_option("-i", "--numepochs", dest="numepochs", default=int(10), help="number of epochs of training; in each epoch we iterate over over training examples")

    (opts, _) = optparser.parse_args()
    f_data = "%s.%s" % (os.path.join(opts.datadir, opts.fileprefix), opts.french)
    e_data = "%s.%s" % (os.path.join(opts.datadir, opts.fileprefix), opts.english)

    if opts.logfile:
        logging.basicConfig(filename=opts.logfile, filemode='w', level=logging.INFO)

    sys.stderr.write("Training with Dice's coefficient...")

    # bitext is a list of list of the e-f pair of words in e-f sentence pair
    # [['sans', 'tout', 'cela', ',', 'nous', 'ne', 'aurons', 'jamais', 'la', 'possibilit\xc3\xa9', 'de', 'voir', 'refl\xc3\xa9t\xc3\xa9s', 'nos', 'espoirs', 'et', 'nos', 'r\xc3\xaaves', '.'], ['without', 'this', 'we', 'will', 'never', 'get', 'that', 'opportunity', 'to', 'see', 'our', 'hopes', 'and', 'dreams', 'reflected', '.']]
    bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:opts.num_sents]]

    Lexical_Aligner(bitext, int(opts.numepochs))
