#!/usr/bin/env python
import optparse, sys, os, bleu, random, math
from collections import namedtuple

def computeFeature(nbest, source):
  sourc = {int(index): line for index, line in enumerate(open(source))}
  for line in open(nbest):
    # ith french sentence, english translated sentence, feature weights
    (i, sentence, features) = line.strip().split("|||")
    features = [float(h) for h in features.strip().split()]
    systems = sentence.strip().split()
    system = set(systems)
    source = set(sourc[int(i.strip())].strip().split())
    diff = " "+str(len(source.difference(system)))
    features.append(float(diff))
    # # print i, "|||", " ".join([str(word) for word in systems]), "|||", " ".join([str(weight) for weight in features])
    lines = line.strip()
    lines+=diff
    print lines



if __name__ == "__main__":
  optparser = optparse.OptionParser()
  optparser.add_option("-n", "--nbest", dest="nbest", default=os.path.join("data", "train.nbest"), help="N-best file")
  optparser.add_option("-r", "--reference", dest="reference", default=os.path.join("data", "train.en"),help="English reference sentences")
  optparser.add_option("-s", "--source", dest="source", default=os.path.join("data", "train.fr"),help="French source sentences")
  (opts, _) = optparser.parse_args()

  output = computeFeature(opts.nbest, opts.source)