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

	def __init__(self, filename, sep='\t', N=None, missingfn=None):
		self.maxlen = 0
		for line in file(filename):
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
		unigram = key.split()[0]
		if key in self: return float(self[key]) / float(self.N)
		elif len(key) == 1: return self.missingfn(key, self.N)
		else: return None

class Entry(object):
	def __init__(self, word, startpos, logprob, backpointer):
		self.word = word
		self.startpos = startpos
		self.logprob = logprob
		self.backpointer = backpointer

Pw  = Pdist(opts.counts2w)
sys.stdout = codecs.lookup('utf-8')[-1](sys.stdout)
# list1 = []
list_sorted = []
for line in file(opts.counts2w):
    bigram = line.split('\t')
    utf8key = unicode(bigram[0], 'utf-8')
    list_sorted.append([utf8key, int(bigram[1])])

for line in list_sorted:
    line.append(np.log2(Pw.__call__(line[0])))
# list_sorted = sorted(list1, key=lambda prob: prob[2], reverse=True)

def checkBigram(utf8line, newindex, objentry, prob):
	for newword in list_sorted:
		bigr = newword[0].split()
		bigram = newword[0].replace(" ", "")
		if bigram in utf8line and utf8line.startswith(bigram, newindex, newindex + len(bigram)):
			newentry = Entry(newword[0], newindex, np.sum([prob, newword[2]]), objentry)
			if newentry not in heap:
				heap.append(newentry)

def checkSingleWord(newindex, oneword, objentry, prob):
	if newindex <= len(oneword) - 1:
		begin = [u'<S>']
		begin.append(oneword[newindex])
		if newindex!=3 and oneword[newindex] not in heap:
			heap.append(Entry(oneword[newindex], newindex, np.sum([prob, np.log2(Pw.__call__(oneword[newindex]))]), objentry))
		elif newindex==3 and " ".join(begin) not in heap:
			heap.append(Entry(" ".join(begin), 0, np.log2(Pw.__call__(oneword[newindex])), None))

def bestSeg(utf8line, chart):
	finalindex = len(utf8line) - 1
	entry = chart[finalindex]
	bestsegmentation = []
	while entry is not None:
		stri = entry.word.split()
		if stri[0] != u'<S>':
			bestsegmentation.append(entry.word)
		else:
			bestsegmentation.append(stri[1])
		entry = entry.backpointer
	bestsegmentation.reverse()
	print " ".join(bestsegmentation)

with open(opts.input) as f:
  for line in f:
	  beg = [u'<S>']
	  heap = []
	  beg.append(unicode(line.strip(), 'utf-8'))
	  utf8line = "".join(beg)
	  chart = [None] * len(utf8line)
	  oneword = [i for i in utf8line]
	  checkBigram(utf8line, 0, None, 0)
	  checkSingleWord(3, oneword, None, 0)
	  while len(heap)!=0:
		  objentry = heap[0]
		  word = heap[0].word
		  prob = heap[0].logprob
		  startpos = heap[0].startpos
		  heap.remove(heap[0])
		  endindex = startpos + len(word.replace(" ", "")) - 1
		  if chart[endindex] is not None:
			  if prob > chart[endindex].logprob:
				  chart[endindex] = objentry
			  else:
				  continue
		  else:
			  chart[endindex] = objentry
		  newindex = endindex + 1
		  checkBigram(utf8line, newindex, objentry, prob)
		  checkSingleWord(newindex, oneword, objentry, prob)
	  bestSeg(utf8line, chart)