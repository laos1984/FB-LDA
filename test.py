import sys, os, re
from cPickle import load, dump
from termcolor import colored
import numpy as np
from fblda import *
import math

def info(s):
  print colored(s, 'yellow')

def readDocs(directory):
  docs = []
  pattern = re.compile('[\W_]+')
  for root, dirs, files in os.walk(directory):
    for filename in files:
      #print colored(filename, 'red')
      with open(root + '/' + filename) as f:
        content = []
        for line in f:
          #print line
          words = [pattern.sub('', w.lower()) for w in line.split()]
          content.extend(words)        
        docs.append(content)
  return docs

def preprocess(directory):
  info('Reading corpus')
  docs_f = readDocs(directory+"/Foreground")
  docs_b = readDocs(directory+"/Background")
  stopwords = load(open('stopwords.pickle'))

  info('Building vocab')
  vocab = set()
  for doc in docs_f:
    for w in doc:
      if len(w) > 1 and w not in stopwords:
        vocab.add(w)
  for doc in docs_b:
    for w in doc:
      if len(w) > 1 and w not in stopwords:
        vocab.add(w)
  vocab       = list(vocab)
  lookupvocab = dict([(v, k) for (k, v) in enumerate(vocab)])

  info('Building BOW representation')
  m_f = np.zeros((len(docs_f), len(vocab)))
  for d, doc in enumerate(docs_f):
    for w in doc:
      if len(w) > 1 and w not in stopwords:
        m_f[d, lookupvocab[w]] += 1
  m_b = np.zeros((len(docs_b), len(vocab)))
  for d, doc in enumerate(docs_b):
    for w in doc:
      if len(w) > 1 and w not in stopwords:
        m_b[d, lookupvocab[w]] += 1
  return (m_f, m_b, vocab)


def discoverTopics(n = 20):
  n_f_topics=20
  n_b_topics=20
  (matrix_f, matrix_b, vocab) = preprocess('./sigir_data/')
  
  sampler = FBLDASampler(n_f_topics, n_b_topics)

  info('Starting!')
  for it, (phi_f, phi_b) in enumerate(sampler.run(matrix_f, matrix_b, 100)):
      print colored("Iteration %s" % it, 'yellow')
      #print "Likelihood", sampler.loglikelihood()
      if it%10==0:
        info('Foreground Topics:')
        for topicNum in xrange(n):
          s = colored(topicNum, 'green')
          words = [(proba, w) for (w, proba) in enumerate(phi_f[topicNum, :]) if proba > 0]
          words = sorted(words, reverse = True)
          for i in range(10):
            proba, w = words[i]
            s += ' ' + vocab[w]
          print s
        info('Background Topics:')
        for topicNum in xrange(n):
          s = colored(topicNum, 'green')
          words = [(proba, w) for (w, proba) in enumerate(phi_b[topicNum, :]) if proba > 0]
          words = sorted(words, reverse = True)
          for i in range(10):
            proba, w = words[i]
            s += ' ' + vocab[w]
          print s

discoverTopics()