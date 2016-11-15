#!/usr/bin/env python
import optparse
import sys
import models
from collections import namedtuple
from math import log10

# Finding the appropriate beam stack with
# value sorted according to their probabilities

def find_beam(stack):
  beam = sorted(stack.itervalues(), key=lambda h: -(h.logprob+ h.f_cost))
  max = beam[0].logprob + beam[0].f_cost
  beam_width = log10(0.000000001)
  beam_filtered = [hypothesis for hypothesis in beam if (hypothesis.logprob+hypothesis.f_cost) >= (max + beam_width)]
  return beam_filtered

# Precomputing future cost estimation for every french sentence
def precompute_future_cost(f, tm):
  fcost = [[0 for _ in f] for _ in f]
  for ln in xrange(1, len(f)+1):
    for start in xrange(0,len(f)-ln+1):
      end = start + ln
      fcost[start][end-1] = -1*sys.maxint
      if f[start:end] in tm:
        e_phrase = max(tm[f[start:end]], key=lambda h: h.logprob)
        fcost[start][end-1] = max(tm[f[start:end]], key=lambda h: h.logprob).logprob
        lm_state = ("<s>",)
        # Computing the language probability for english phrase
        for word in e_phrase.english.split():
          (lm_state, word_logprob) = lm.score(lm_state, word)
          fcost[start][end - 1] += word_logprob

      for ind in xrange(start, end-1):
        if fcost[start][ind] + fcost[ind+1][end-1] > fcost[start][end-1]:
          fcost[start][end-1] = fcost[start][ind] + fcost[ind+1][end-1]

  return fcost



# The following code implements a monotone decoding

def beam_stack_decode(french, tm, lm, opts):

  sys.stderr.write("Decoding %s...\n" % (opts.input,))

  for s_no,f in enumerate(french):
    sys.stderr.write("French Sentence --> %s...\n" % str(int(s_no) + 1))

    # Tweaking the params based on len of french sen
    o_eta = opts.eta
    o_s  = opts.s
    o_distort = opts.distort
    if len(f) < 9:
      o_eta = 0.9
      o_s = 3000
      o_distort = 6

    if len(f) > 14:
      o_eta = 0.9
      o_s  = 3000
      o_distort = 6

    # Pre-computing the future_cost for f
    f_cost = precompute_future_cost(f, tm)

    bit_vec = [0]*len(f)

    # Initialising the hypothesis
    hypothesis = namedtuple("hypothesis", "logprob, lm_state, bit_vec, last_ind, predecessor, phrase, f_cost")
    initial_hypothesis = hypothesis(0.0, lm.begin(), bit_vec, 0, None, None, 0) #f_cost[0][len(f)-1])

    #Creating the stacks
    stacks = [{} for _ in f] + [{}]

    bitHash = ''.join(str(bit) for bit in bit_vec)
    stacks[0][bitHash]= initial_hypothesis

    # Iterating over all stacks except the one where all words
    # are translated
    for i, stack in enumerate(stacks[:-1]):

      # Find the sorted and pruned stack
      beam = find_beam(stack)

      # Iterating on Pruned Hypotheses - Histogram Pruning
      for h in beam[:o_s]:

        # Iterating over all phrase possibilities
        for x in xrange(0,len(f)):
          for y in xrange(x+1,len(f)+1):

            if 1 in h.bit_vec[x:y]:
              continue

            if i == 0 and abs(1 - x) > o_distort:
              break

            f_phrase = f[x:y]
            if f_phrase in tm:

              # Deep copying the bit vector
              new_bit_vec = h.bit_vec[:]
              for bt in xrange(x,y):
                new_bit_vec[bt] = 1

              for phrase in tm[f_phrase]:

                # Adding the phrase translation probability
                logprob = h.logprob + phrase.logprob
                lm_state = h.lm_state

                # Computing the language probability for english phrase
                for word in phrase.english.split():
                  (lm_state, word_logprob) = lm.score(lm_state, word)
                  logprob += word_logprob
                if 0 not in new_bit_vec:
                  logprob += lm.end(lm_state)
                if abs(h.last_ind + 1 - x) > o_distort:
                  logprob += log10(o_eta) * abs(h.last_ind + 1 - x)


                # Calculate the future cost of untranslated words
                st = -1
                fcost = 0
                for bit in xrange(0,len(new_bit_vec)):
                  if st == -1 and new_bit_vec[bit] == 0:
                    st = bit
                  if st != -1 and new_bit_vec[bit] == 1:
                    fcost += f_cost[st][bit-1]
                    st = -1
                if st != -1:
                    fcost += f_cost[st][len(new_bit_vec) -1]

                # Check for the correct stack number
                if i+y-x != new_bit_vec.count(1):
                    sys.stderr.write("The stack number is not correct i+y-x = %s and bit_vec_count = %s \n" % (i+y-x, new_bit_vec.count(1)))

                # Create the new hypothesis
                new_hypothesis = hypothesis(logprob, lm_state, new_bit_vec, y-1, h, phrase, fcost)

                # Recombination in the
                bitHash = ''.join(str(bit) for bit in new_bit_vec)
                if bitHash not in stacks[i+y-x] or stacks[i+y-x][bitHash].logprob < logprob: # second case is recombination
                  stacks[i+y-x][bitHash] = new_hypothesis

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


if __name__=="__main__":
  optparser = optparse.OptionParser()
  optparser.add_option("-i", "--input", dest="input", default="data/input", help="File containing sentences to translate (default=data/input)")
  optparser.add_option("-t", "--translation-model", dest="tm", default="data/tm", help="File containing translation model (default=data/tm)")
  optparser.add_option("-l", "--language-model", dest="lm", default="data/lm", help="File containing ARPA-format language model (default=data/lm)")
  optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to decode (default=no limit)")
  optparser.add_option("-k", "--translations-per-phrase", dest="k", default=1, type="int", help="Limit on number of translations to consider per phrase (default=1)")
  optparser.add_option("-s", "--stack-size", dest="s", default=1, type="int", help="Maximum stack size (default=1)")
  optparser.add_option("-e", "--eta", dest="eta", default=0.6, type="float", help="Eta Value for distortion model (default=0.6)")
  optparser.add_option("-d", "--distort", dest="distort", default=4, type="int", help="Maximum distortion length (default=4)")
  optparser.add_option("-a", "--alpha", dest="alpha", default=0.0001, type="float", help="Alpha value for threshold pruning (default=0.0001)")
  optparser.add_option("-v", "--verbose", dest="verbose", action="store_true", default=False,  help="Verbose mode (default=off)")
  opts = optparser.parse_args()[0]

  tm = models.TM(opts.tm, opts.k)
  lm = models.LM(opts.lm)

  # Getting the French sentences
  french = [tuple(line.strip().split()) for line in open(opts.input).readlines()[:opts.num_sents]]

  # tm should translate unknown words as-is with probability 1
  for word in set(sum(french,())):
    if (word,) not in tm:
      tm[(word,)] = [models.phrase(word, 0.0)]

  beam_stack_decode(french, tm, lm, opts)