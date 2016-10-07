"""
You have to write the perc_train function that trains the feature weights using the perceptron algorithm for the CoNLL 2000 chunking task.
Each element of train_data is a (labeled_list, feat_list) pair.
Inside the perceptron training loop:
    - Call perc_test to get the tagging based on the current feat_vec and compare it with the true output from the labeled_list
    - If the output is incorrect then we have to update feat_vec (the weight vector)
    - In the notation used in the paper we have w = w_0, w_1, ..., w_n corresponding to \phi_0(x,y), \phi_1(x,y), ..., \phi_n(x,y)
    - Instead of indexing each feature with an integer we index each feature using a string we called feature_id
    - The feature_id is constructed using the elements of feat_list (which correspond to x above) combined with the output tag (which correspond to y above)
    - The function perc_test shows how the feature_id is constructed for each word in the input, including the bigram feature "B:" which is a special case
    - feat_vec[feature_id] is the weight associated with feature_id
    - This dictionary lookup lets us implement a sparse vector dot product where any feature_id not used in a particular example does not participate in the dot product
    - To save space and time make sure you do not store zero values in the feat_vec dictionary which can happen if \phi(x_i,y_i) - \phi(x_i,y_{perc_test}) results in a zero value
    - If you are going word by word to check if the predicted tag is equal to the true tag, there is a corner case where the bigram 'T_{i-1} T_i' is incorrect even though T_i is correct.
"""

import perc
import sys, optparse, os, random
from collections import defaultdict

class Chunker(object):
    def __init__(self):
        self.counter = 0
        self.feat_vec_cache = defaultdict(int)
        self.coun_dict = defaultdict(int)

    def perc_train(self, train_data, tagset, numepochs):
        # perceptron train
        # feat_vec stores the weights for the features of a sentence, initially all weights are 0
        feat_vec = defaultdict(int)
        # default_tag = 'B-NP'
        default_tag = tagset[0]
        # for each epoch/iteration
        for i in range(0, numepochs):
            # for each item (e.g tuple=([labeled words for each sentence],[features for those words of sentence])) in train_data
            for (label_list, feat_list) in train_data:
                # cur = list of best tag for each word in sentence found using viterbi algo
                cur = perc.perc_test(feat_vec, label_list, feat_list, tagset, default_tag)
                # gold = list of reference/true tag for each word in sentence
                gold = [entry.split()[2] for entry in label_list]
                if cur != gold:
                    self.counter+=1
                    cur.insert(0, 'B_-1')
                    gold.insert(0, 'B_-1')
                    cur.append('B_+1')
                    gold.append('B_+1')
                    cur_len = len(cur)
                    gold_len = len(gold)
                    if cur_len != gold_len:
                        raise ValueError("output length is not the same with the input sentence")
                    feat_index = 0
                    # perceptron update
                    # for each tag/word of a sentence
                    for i in range(1, cur_len):
                        # for each word in a sentence, (feat_index, features) is a tuple, where feat_index=endindex of the list of features for that word, and features=list of features for that word
                        (feat_index, features) = perc.feats_for_word(feat_index, feat_list)
                        # update the weights of the features for that word, by rewarding the features seen in reference, while penalizing the ones not seen in reference but returned by viterbi
                        for f in features:
                            self.computeCacheVector(cur[i], f, feat_vec, -1.0)
                            self.computeCacheVector(gold[i], f, feat_vec, 1.0)
            print >>sys.stderr, "iteration %d done."%i
        self.computeAvgWeights()
        return self.feat_vec_cache

    def computeCacheVector(self, label, feature, feat_vec, val):
        self.feat_vec_cache[(feature,label)] = self.feat_vec_cache[(feature,label)] + (self.counter - self.coun_dict[(feature,label)]) * feat_vec[(feature,label)]
        self.coun_dict[(feature, label)] = self.counter
        feat_vec[(feature,label)] = feat_vec[(feature,label)] + val

    def computeAvgWeights(self):
        for key, value in self.feat_vec_cache.items():
            (feature, label) = key
            self.feat_vec_cache[(feature, label)] = round(self.feat_vec_cache[(feature, label)]) / float(self.counter)
        return self.feat_vec_cache

if __name__ == '__main__':
    optparser = optparse.OptionParser()
    optparser.add_option("-t", "--tagsetfile", dest="tagsetfile", default=os.path.join("data", "tagset.txt"), help="tagset that contains all the labels produced in the output, i.e. the y in \phi(x,y)")
    optparser.add_option("-i", "--trainfile", dest="trainfile", default=os.path.join("data", "train.txt.gz"), help="input data, i.e. the x in \phi(x,y)")
    optparser.add_option("-f", "--featfile", dest="featfile", default=os.path.join("data", "train.feats.gz"), help="precomputed features for the input data, i.e. the values of \phi(x,_) without y")
    optparser.add_option("-e", "--numepochs", dest="numepochs", default=int(1), help="number of epochs of training; in each epoch we iterate over over all the training examples")
    optparser.add_option("-m", "--modelfile", dest="modelfile", default=os.path.join("data", "default.model"), help="weights for all features stored on disk")
    (opts, _) = optparser.parse_args()

    # each element in the feat_vec dictionary is:
    # key=feature_id value=weight
    feat_vec = {}
    tagset = []
    train_data = []

    # tagset contains list of the tags in tagset.txt
    # ['B-NP', 'I-NP', 'O', 'B-VP', 'B-PP', 'I-VP', 'B-ADVP', 'B-SBAR', 'B-ADJP', 'I-ADJP', 'B-PRT', 'I-ADVP', 'I-PP', 'I-CONJP', 'I-SBAR', 'B-CONJP', 'B-INTJ', 'B-LST', 'I-INTJ', 'I-UCP', 'I-PRT', 'I-LST', 'B-UCP']
    tagset = perc.read_tagset(opts.tagsetfile)
    # print 'tagset', tagset
    print >>sys.stderr, "reading data ..."
    # opts.trainfile contains the labeled training data with word, POS, chunk tag in each line
    # opts.featfile contain unigram and bigram features with their feature id
    # train_data is a list of 8936 tuples, where each tuple=([list of labeled words making up each sentence],[list of features for those words of that sentence])
    # (['Confidence NN B-NP', 'in IN B-PP', 'the DT B-NP', 'pound NN I-NP', 'is VBZ B-VP', 'widely RB I-VP','expected VBN I-VP', 'to TO I-VP', 'take VB I-VP', 'another DT B-NP', 'sharp JJ I-NP', 'dive NN I-NP',.....,'. . O'], ['U00:_B-2', 'U01:_B-1', 'U02:Confidence', 'U03:in', 'U04:the', 'U05:_B-1/Confidence', 'U06:Confidence/in',......, 'U17:./_B+1', 'U18:_B+1/_B+2', 'U20:JJ/NNS/.', 'U21:NNS/./_B+1', 'U22:./_B+1/_B+2', 'B'])
    train_data = perc.read_labeled_data(opts.trainfile, opts.featfile)
    # print 'train_data', type(train_data), len(train_data)
    # for i in train_data:
    #     print type(i), len(i), i
    print >>sys.stderr, "done."
    c = Chunker()
    feat_vec = c.perc_train(train_data, tagset, int(opts.numepochs))
    perc.perc_write_to_file(feat_vec, opts.modelfile)