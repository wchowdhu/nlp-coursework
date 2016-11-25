#!/usr/bin/env python
import optparse, sys, os, bleu, random, math
from collections import namedtuple

# samples generated from n-best list per input sentence
tau = 5000
# sampler acceptance cutoff
alpha = 0.1
# training data generated from the samples tau
xi = 100
# perceptron learning rate
eta = 0.1
# number of epochs for perceptron training
epochs = 5
# lines = [index for index,line in enumerate(open(opts.reference))]
# nbests = [[] for x in xrange(lines[-1]+1)]
nbests = [[] for x in xrange(1989)]

def computeBleu(system, reference):
  stats = [0 for i in xrange(10)]
  stats = [sum(scores) for scores in zip(stats, bleu.bleu_stats(system,reference))]
  return bleu.smoothed_bleu(stats)

def get_sample(nbest):
  # sample contains a list of tuples for a source sentence, where each tuple = (s1,s2)
  sample = []
  if len(nbest) >= 2:
    for i in range(tau):
      # select returns a list of 2 randomly selected items from nbest
      select = random.sample(nbest, 2)
      s1 = select[0]
      s2 = select[1]
      if math.fabs(s1.smoothed_bleu - s2.smoothed_bleu) > alpha:
        if s1.smoothed_bleu > s2.smoothed_bleu:
          # sample += (s1, s2)
          sample.append((s1, s2))
        else:
          # sample += (s2, s1)
          sample.append((s2, s1))
      else:
        continue
  else:
    s1 = nbest[0]
    s2 = nbest[0]
    sample.append((s1, s2))
  return sample

def computeNBests():
  ref = {int(index): line for index, line in enumerate(open(opts.reference))}
  for line in open(opts.nbest):
    # ith french sentence, english translated sentence, feature weights
    (i, sentence, features) = line.strip().split("|||")
    system = sentence.strip().split()
    reference = ref[int(i.strip())].strip().split()
    score = computeBleu(system, reference)
    features = [float(h) for h in features.strip().split()]
    translation = namedtuple("translation", "sentence, smoothed_bleu, features")
    nbests[int(i.strip())].append(translation(sentence, score, features))
    # theta = [float(1.0) for _ in xrange(len(features))]
    theta = [random.uniform(-1*sys.maxint, 1.0) for _ in xrange(len(features))]
  return [nbests,theta]

def computePRO(nbests,theta):
  for i in range(epochs):
    mistakes = 0
    # nbest contains a list of (english translation, bleu score, features) tuples for a source sentence
    for nbest in nbests:
      sample = get_sample(nbest)
      # sort the tau samples from get_sample() using s1.smoothed_bleu - s2.smoothed_bleu
      sample.sort(key=lambda tup: math.fabs(tup[0].smoothed_bleu - tup[1].smoothed_bleu))
      sample.reverse()
      # keep the top xi (s1, s2) values from the sorted list of samples
      sample = sample[:xi]
      # do a perceptron update of the parameters theta:
      for tup in sample:
        s1,s2 = tup
        # if theta * s1.features <= theta * s2.features:
        if sum([x * y for x, y in zip(theta, s1.features)]) <= sum([x * y for x, y in zip(theta, s2.features)]):
          mistakes += 1
          # theta += eta * (s1.features - s2.features)  # this is vector addition!
          # update = eta * (s1.features - s2.features)
          update = [item * eta for item in [x - y for x, y in zip(s1.features, s2.features)]]
          # theta = theta + update
          theta = [x + y for x, y in zip(theta, update)]
          # sys.stderr.write(" ".join([str(weight) for weight in theta]))
          # sys.stderr.write("\n")
    sys.stderr.write("Mistakes --> %s...\n" % str(int(mistakes)))
  return theta


if __name__ == "__main__":
  optparser = optparse.OptionParser()
  optparser.add_option("-n", "--nbest", dest="nbest", default=os.path.join("data", "train.nbest"), help="N-best file")
  optparser.add_option("-r", "--reference", dest="reference", default=os.path.join("data", "train.en"),help="English reference sentences")
  (opts, _) = optparser.parse_args()

  output = computeNBests()
  weights = computePRO(output[0],output[1])
  print "\n".join([str(weight) for weight in weights])
