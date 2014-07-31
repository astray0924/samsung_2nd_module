# coding: utf-8
from __future__ import division
import os
import nltk
import cPickle as pickle
import numpy as np
import re
import codecs
import collections
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

  def train(self, train_file='resources/big.txt'):
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

class Vectorizer:
    def __init__(self):
        # 분석할 데이터
        self.data = None

        # 벡터화에 필요한 도움 클래스
        # spelling corrector
        self.corrector = SpellingCorrector()
        self.corrector.train()

        # stopword
        self.stopwords = set(nltk.corpus.stopwords.words('english'))

        # Lemmatizer
        self.lmtzr = WordNetLemmatizer()

        # sent pattern
        self.sent_pattern = re.compile(r'\b(\w+/NN\s?)+\w+/VB[PZ]? (\w+/RB\s)?\w+/JJ(\s*,/, (\w+/RB\s)?\w+/JJ(\s,/,)?)*(\s\w+/CC (\w+/RB\s)?\w+/JJ)?\s')

        # NP 및 형용사 추출을 위한 패턴
        # TODO: Stanford 툴킷에서 출력하는 POS 태그와 호환되는지 체크할 필요 있음
        RE_FLAGS = re.UNICODE
        self.np_pattern = re.compile(r'\w+/NN[PS]{0,2}', RE_FLAGS)
        self.adj_pattern = re.compile(r'\w+/JJ[RS]{0,1}', RE_FLAGS)
        self.tag_pattern = re.compile(r'/\w+', RE_FLAGS)

        # NLTK Chunker
        NP_GRAMMAR = """
            NP:   {<NN.*>+}
        """
        self.np_chunker = nltk.RegexpParser(NP_GRAMMAR)

        # 결과물
        self.np_vectors = None
        self.np_counter = None

    def pos_sent_to_tuples(self, pos_sent):
        """
        POS 태깅된 문자를 받아서 단어-태그 형태의 튜플 리스트 형태로 반환
        단어는 소문자화 함
        """
        try:
            return [(wt.split('/')[0], wt.split('/')[1]) 
                               for wt in pos_sent.strip().split()]
        except IndexError:
            return []

    def tuples_to_pos_sent(self, tuples):
        return ' '.join(['/'.join(t) for t in tuples])

    def extract_all_np(self, pos_sent):
        tags = self.pos_sent_to_tuples(pos_sent)
        
        try:
            t = self.np_chunker.parse(tags)
        except Exception:
            return []
        
        nps = []
        self._extract_np(t, nps)
        return nps

    def _extract_np(self, t, nps=[]):
        try:
            t.node
        except AttributeError:
            return None
        else:
            if t.node == 'NP':
                np = t.pprint().replace('(NP ', '').replace(')', '').strip()
                nps.append(np)
            else:
                for child in t:
                    self._extract_np(child, nps)
                
    def extract_all_jj(self, pos_sent):
        return re.findall(self.adj_pattern, pos_sent)

    def extract_features(self, pos_sent):
        np_context = defaultdict(list)
        all_np = self.extract_all_np(pos_sent)
        all_jj = self.extract_all_jj(pos_sent)
        
        for np in all_np:
            np = re.sub(self.tag_pattern, '', np)
            np_context[np] += [re.sub(self.tag_pattern, '', tag) for tag in all_jj]

        return np_context

    # 패턴에 맞는 문장들을 추출
    def _extract_pattern_sents(self, data):
        for i, line in enumerate(data):
            line = line.strip()
            if not line:
                continue

            sent_match = re.search(self.sent_pattern, line)
            if sent_match:
                yield sent_match.group(0)

    def vectorize(self, data_file, encoding='utf-8'):
        """
        POS 태깅된 텍스트를 받아서 그것에 포함된 NP들의 벡터를 생성함.
        """
        try:
            self.data = codecs.open(data_file, encoding=encoding)
            self.data.seek(0)
        except IOError, e:
            raise IOError(e)

        # 데이터 파일이 유효하지 않다면 에러 출력
        if self.data is None:
            raise AttributeError("Cannot load the data")

        # 데이터에서 NP 및 feature 추출
        self.tokens = []
        self.np_context = defaultdict(Counter)
        self.np_counter = Counter()

        for i, line in enumerate(self._extract_pattern_sents(self.data)):
            # 해당 줄이 비어있으면 continue
            line = line.strip()
            if not line:
                continue

            # 줄 단위로 처리
            for np, context in self.extract_features(line).iteritems():
                # 언어적 전처리 수행 (NP)
                np = np.lower()
                # np = self.corrector.correct(np)
                np = self.lmtzr.lemmatize(np, pos='n')

                # 언어적 전처리 수행 (ADJ)
                context = [self.corrector.correct(c.lower()) for c in context]

                # 중간 결과물 저장
                self.np_counter.update([np])
                self.np_context[np].update(context)
                self.tokens += [np]
                self.tokens += context

            # 전체 토큰의 개수를 저장
            self.token_counter = Counter()
            self.token_counter.update(self.tokens)

            # PPMI 계산
            self.np_context_ppmi = dict()
            normalizing_comp = sum(self.token_counter.values())

            for np, context in self.np_context.iteritems():
                # p_n: text에서 해당 명사가 출현할 확률
                p_n = self.token_counter[np] / normalizing_comp
                
                # p_a: text에서 해당 형용사가 출현할 확률
                # p_na: text에서 해당 명사와 형용사가 동시에 출현할 확률
                # pmi: log( p_na / (p_n * p_a)  )
                context_ppmi = dict()
                for adj, count in context.iteritems():
                    p_a = self.token_counter[adj] / normalizing_comp
                    p_na = count / normalizing_comp
                    
                    # 처리중인 형용사의 PPMI 값 계산
                    pmi = log( p_na / (p_n * p_a) )
                    ppmi = pmi if pmi > 0 else 0
                    
                    # 형용사의 PPMI 값을 dict 형태로 저장
                    context_ppmi[adj] = ppmi
                    
                # 명사 context의 PPMI 값 저장
                self.np_context_ppmi[np] = context_ppmi

            # NumPy 벡터로 변환
            # DictVectorizer 사용
            self.vectorizer = DictVectorizer()
            self.vectorizer.fit(self.np_context_ppmi.values())
            self.np_vectors = dict()
            for np, context in self.np_context_ppmi.iteritems():
                self.np_vectors[np] = self.vectorizer.transform(context).tocsr() 

    def get_vector(self, np):
        if self.np_vectors is None:
            raise AttributeError("You must first vectorize the data by calling vectorize(<data_file>)")

        if np in self.np_vectors:
            return self.np_vectors[np]
        else:
            return None

    def get_top_nps(self, k=100):
        return self.np_counter.most_common(k)

    def dump_vectors(self, dir='./vectors'):
        raise NotImplementedError

if __name__ == '__main__':
    v = Vectorizer()
    v.vectorize('resources/sample_data.txt')
    print v.get_top_nps()