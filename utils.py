# coding: utf-8
from __future__ import division
import os
import nltk
import cPickle as pickle
import numpy as np
import re
import codecs
import collections
from spelling_corrector import SpellingCorrector
from math import log
from nltk.corpus import stopwords
from collections import Counter
from collections import defaultdict
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize  
from pprint import pprint
from sklearn import cluster
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pprint import pprint

class SpellingCorrector:
  def __init__(self):
    self.alphabet = 'abcdefghijklmnopqrstuvwxyz'
    self.NWORDS = None

  def words(self, text):
    return re.findall('[a-z]+', text.lower()) 

  def train(self, train_file='big.txt'):
    self.NWORDS = self._train(self.words(file(train_file).read()))

  def _train(self, features):
    model = collections.defaultdict(lambda: 1)
    for f in features:
      model[f] += 1
    return model

  def edits1(self, word):
    splits     = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes    = [a + b[1:] for a, b in splits if b]
    transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b)>1]
    replaces   = [a + c + b[1:] for a, b in splits for c in self.alphabet if b]
    inserts    = [a + c + b     for a, b in splits for c in self.alphabet]
    return set(deletes + transposes + replaces + inserts)

  def known_edits2(self, word):
    return set(e2 for e1 in self.edits1(word) for e2 in self.edits1(e1) if e2 in self.NWORDS)

  def known(self, words): 
    return set(w for w in words if w in self.NWORDS)

  def correct(self, word):
    if not self.NWORDS:
      raise AttributeError("The corrector must be trained by calling train()")

    candidates = self.known([word]) or self.known(self.edits1(word)) or self.known_edits2(word) or [word]
    return max(candidates, key=self.NWORDS.get)

class LemmaTokenizer(object):
     def __init__(self):
        self.wnl = WordNetLemmatizer()
        self.pattern = re.compile(r'\w+')
     def __call__(self, doc):
         return [self.wnl.lemmatize(t) for t in self.pattern.findall(doc)]