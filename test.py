import json
import nltk
import string
import re
import html
from math import log10
import numpy as np
import operator
from collections import Counter
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

stop_words = stopwords.words('english')
stemmer = PorterStemmer()

keys_filename = "data/orgKeys.txt"
names_filename = "data/orgNames.json"
desc_filename = "data/orgDescriptions.json"
summaries_filename = "data/orgSummaries.json"

keys = [line.rstrip('\n') for line in open(keys_filename)]

with open(names_filename) as f:
	nameDict = json.load(f)
with open(desc_filename) as f:
	descriptionDict = json.load(f)
with open(summaries_filename) as f:
	summaryDict = json.load(f)

descriptions = list(descriptionDict.values())

def tokenize(str):
	s = str.lower()
	regx = re.compile('\W+')
	tokens = [t for t in regx.split(s) if len(t)>=3 and t.isalpha() and t not in stop_words]
	stemmed = [stemmer.stem(t) for t in tokens]
	return stemmed

def tokenize_dict(revs):
    id_tokens = {}
    for key,val in revs.items():
        tokens = tokenize(val)
        id_tokens[key] = tokens
    return id_tokens

def ten_most_popular_words(revs):
    all_tokens = []
    for r in revs:
        tokens = tokenize(r)
        all_tokens += tokens
    counts = Counter(all_tokens)
    most_frequent = counts.most_common(10)
    return most_frequent

def raw_tf(s_words, all_words):
    count = []
    for w in all_words:
        count.append(s_words.count(w))
    return count

def log_tf(tf):
    logtf = []
    for f in tf:
        if f != 0:
            logtf.append(1 + log10(f))
        else:
            logtf.append(0)
    return logtf

def idf_t(n, df):
    idf = []
    for f in df:
        if f != 0:
            idf.append(log10(n/f))
        else:
            idf.append(0)
    return idf
        
def sim(q, d):
    dot_prod = np.dot(q,d)
    mag_q = np.linalg.norm(q)
    mag_d = np.linalg.norm(d)
    if mag_q == 0 or mag_d ==0:
        return 0
    return dot_prod/(mag_q * mag_d)

def all_words_ranked(revs):
    all_tokens = []
    for r in revs:
        tokens = tokenize(r)
        all_tokens += tokens
    counts = Counter(all_tokens)
    most_frequent = counts.most_common()
    return most_frequent

def score(query, docs, all_tokens):
    scores = {}
    doc_log_tf = {}
    df = [0] * len(all_tokens)
    q_words = tokenize(query)
    q_counts = raw_tf(q_words,all_tokens)
    
    for docid,doc_tokens in docs.items():
        doc_counts = raw_tf(doc_tokens, all_tokens)
        doc_log_tf[docid] = log_tf(doc_counts)
    
    i = 0
    for t in all_tokens:
        for docid,doc_tokens in docs.items():
            if t in doc_tokens:
                df[i] += 1
        i += 1
    N = len(docs)
    idf = idf_t(N, df)
    
    for docid,doc_tokens in docs.items():
        tfidf = [a*b for a,b in zip(doc_log_tf.get(docid),idf)]
        scor = sim(q_counts,tfidf)
        scores[docid] = scor
       
    sorted_scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
    
    return sorted_scores
    # return sorted_scores[0:10]

if __name__ == '__main__':
	q = "sports"
	words_and_counts = all_words_ranked(descriptions)
	all_tokens = [word for word,count in words_and_counts]
	docs = tokenize_dict(descriptionDict)
	results = score(q, docs, all_tokens)
	i = 1
	for docid,s in results:
		if s == 0:
			break
		print(i, docid, s, "\n")
		i += 1

