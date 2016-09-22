# -*- coding: utf-8 -*-

import sys, codecs, optparse, os, csv, operator, math, numpy as np

# create an instance of the OptionParser class
optparser = optparse.OptionParser()
# Then start defining options using add_options() method
optparser.add_option("-c", "--unigramcounts", dest='counts1w', default=os.path.join('data', 'count_1w.txt'), help="unigram counts")
optparser.add_option("-b", "--bigramcounts", dest='counts2w', default=os.path.join('data', 'count_2w.txt'), help="bigram counts")
optparser.add_option("-i", "--inputfile", dest="input", default=os.path.join('data', 'input'), help="input file to segment")
# Once all of your options are defined, instruct optparse to parse your programs command line using parse_args()
(opts, _) = optparser.parse_args()

class Pdist(dict):
	"A probability distribution estimated from counts in datafile."

	def __init__(self, filename, sep='\t', N=None, missingfn=None):
		# self is a dictionary which will contain as key=words/unigrams and value=frequency with len(self)=9528 total unigrams
		# filename is 'data\\count_1w.txt'
		self.maxlen = 0
		for line in file(filename):
			# in each line, the key=word and value=freq in the file are separated by \t (e.g tab) so we use \t to tokenize each line
			(key, freq) = line.split(sep)
			try:
				utf8key = unicode(key, 'utf-8')
			except:
				raise ValueError("Unexpected error %s" % (sys.exc_info()[0]))
			self[utf8key] = self.get(utf8key, 0) + int(freq)
			self.maxlen = max(len(utf8key), self.maxlen)
		self.N = float(N or sum(self.itervalues()))
		self.missingfn = missingfn or (lambda k, N: 1./N)

	def __call__(self, key):
		# default algorithm didn't use call method
		# float(self[key])/float(self.N) returns the probability (freq of each unigram word/total freq of all words in file) for each key (e.g. unigram word in file) in dictionary
		if key in self: return float(self[key])/float(self.N)
		#else: return self.missingfn(key, self.N)
		# else if the word consists of only a single character (e.g. len=1), then return missingfn=1.0/N (the character occured only once/total frequency of all words in file)
		elif len(key) == 1: return self.missingfn(key, self.N)
		else: return None

class Entry(object):
	def __init__(self, word, startpos, logprob, backpointer):
		self.word = word
		self.startpos = startpos
		self.logprob = logprob
		self.backpointer = backpointer

# the default segmenter does not use any probabilities, but you could ...
Pw  = Pdist(opts.counts1w)
# Standard output and standard error (commonly abbreviated stdout and stderr) are pipes that are built into every UNIX system. When you print something, it goes to the stdout pipe; when your program crashes and prints out debugging information (like a traceback in Python), it goes to the stderr pipe.
# old = sys.stdout
sys.stdout = codecs.lookup('utf-8')[-1](sys.stdout)
list1 = []
for line in file(opts.counts1w):
    list1.append(unicode(line.strip(), 'utf-8'))
list = [line.split() for line in list1]
for line in list:
    line.append(np.log2(Pw.__call__(line[0])))
# unigrams sorted according to logprob in descending order
# list_sorted is a sorted list of lists where each entry=list=[word,freq,prob]
list_sorted = sorted(list, key=lambda prob: prob[2], reverse=True)

with open(opts.input) as f:
  # for each input/sequence of characters
  for line in f:
      heap = []
      utf8line = unicode(line.strip(), 'utf-8')
      chart = [None] * len(utf8line)
      # oneword contain words which are 1 character long, and assign a small freq=1/N to them
      oneword = [i for i in utf8line]
      # for each word in list_sorted that matches input at position 0
      for entry in list_sorted:
          if entry[0] in utf8line and utf8line.startswith(entry[0], 0, len(entry[0])):
              heap.append(Entry(entry[0], 0, entry[2], None))
      if oneword[0] not in heap:
          heap.append(Entry(oneword[0], 0, np.log2(Pw.__call__(oneword[0])), None))
      while len(heap)!=0:
          objentry = heap[0]
          word = heap[0].word
          prob = heap[0].logprob
          startpos = heap[0].startpos
          heap.remove(heap[0])
          endindex = startpos + len(word)-1
          if chart[endindex] is not None:
              if prob > chart[endindex].logprob:
                  chart[endindex] = objentry
              else:
                  continue
          else:
              chart[endindex] = objentry
          newindex = endindex + 1
          for newword in list_sorted:
                if newword[0] in utf8line and utf8line.startswith(newword[0], newindex, newindex + len(newword[0])):
                        newentry = Entry(newword[0], newindex, np.sum([prob, newword[2]]), objentry)
                        if newentry not in heap:
                            heap.append(newentry)
          if newindex <= len(oneword)-1 and oneword[newindex] not in heap:
              heap.append(Entry(oneword[newindex], newindex, np.sum([prob, np.log2(Pw.__call__(oneword[newindex]))]), objentry))

      finalindex = len(utf8line)-1
      entry = chart[finalindex]
      bestsegmentation = []
      while entry is not None:
          bestsegmentation.append(entry.word)
          entry = entry.backpointer
      bestsegmentation.reverse()
      print " ".join(bestsegmentation)
# sys.stdout = old
