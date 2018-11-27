from flask import Flask, render_template, request
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
from scipy import spatial

nltk.download('stopwords')
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

words_and_counts = all_words_ranked(descriptions)
all_tokens = [word for word,count in words_and_counts]
docs = tokenize_dict(descriptionDict)

doc_log_tf = {}
df = [0] * len(all_tokens)

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

tfidf_dict = {}
for docid,doc_tokens in docs.items():
	tfidf_dict[docid] = [a*b for a,b in zip(doc_log_tf.get(docid),idf)]

def score(query, docs, all_tokens):
    scores = {}
    # doc_log_tf = {}
    # df = [0] * len(all_tokens)
    q_words = tokenize(query)
    q_counts = raw_tf(q_words,all_tokens)
    
    # for docid,doc_tokens in docs.items():
    #     doc_counts = raw_tf(doc_tokens, all_tokens)
    #     doc_log_tf[docid] = log_tf(doc_counts)
    
    # i = 0
    # for t in all_tokens:
    #     for docid,doc_tokens in docs.items():
    #         if t in doc_tokens:
    #             df[i] += 1
    #     i += 1
    # N = len(docs)
    # idf = idf_t(N, df)
    
    for docid,doc_tokens in docs.items():
        # tfidf = [a*b for a,b in zip(doc_log_tf.get(docid),idf)]
        tfidf = tfidf_dict[docid]
        scor = sim(q_counts,tfidf)
        scores[docid] = scor
       
    sorted_scores = sorted(scores.items(), key=operator.itemgetter(1), reverse=True)
    
    return sorted_scores
    # return sorted_scores[0:10]

# function that uses cos sim to get
# the similar documents.
def similar_orgs(docID,docs,tokens_and_counts):
	scores = {}
	for doc,tokens in docs.items():
		if doc == docID:
			continue
		doc1vec = []
		dec2vec = []
		#print(docs[docID])
		#print(tokens)
		for word in docs[docID]:
			if word in tokens:
				doc1vec.append(1)
				dec2vec.append(1)
			## below we take into account the frequency of the words
			## words more common will effect the score less
			else:
				doc1vec.append(1*(1/log10(tokens_and_counts[word]+1)))
				dec2vec.append(0)
		for word in tokens:
			if word not in docs[docID]:
				doc1vec.append(0)
				dec2vec.append(1*(1/log10(tokens_and_counts[word]+1)))
		scores[doc] = 1- spatial.distance.cosine(doc1vec,dec2vec)
	
	return sorted(scores.items(), key=operator.itemgetter(1),reverse =True)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/results',methods=['GET', 'POST'])
def results():
    if(request.method == 'POST'):
        # words_and_counts = all_words_ranked(descriptions)
        # all_tokens = [word for word,count in words_and_counts]
        # docs = tokenize_dict(descriptionDict)

        q = request.form['query']
        # q = "engineering"

        # print("\nSearch results for query:", q, "\n")
        results = score(q, docs, all_tokens)
        #return render_template(results,'index.html')
        testid = ""
        i = 1
        for docid,s in results:
            if s == 0:
                break
            # print(i, docid, s)
            print(i, nameDict[docid], s)
            i += 1
        print("\n", i-1, "results found!\n\n")
        return render_template('results.html', results = results, dictionary = nameDict, query = q, numresults = i-1)

@app.route('/organization',methods=['GET', 'POST'])
def organization():
    if(request.method == 'POST'):
        # words_and_counts = all_words_ranked(descriptions)
        # all_tokens = [word for word,count in words_and_counts]
        # docs = tokenize_dict(descriptionDict)
        org = request.form['organization']
        num_recommendations = 10
        print("Top", num_recommendations ,"organizations similar to:", nameDict[org], "\n")
        word_dict = dict(words_and_counts)
        results = similar_orgs(org,docs,word_dict)
        results2 = []
        i = 1
        for docid,s in results:
            if i >= num_recommendations+1:
                break
            # print(i, docid, s)
            print(i, nameDict[docid], s)
            results2.append(results[i])
            i += 1
        print("\n")
        return render_template('organization.html', organization = org, results = results2, dictionary = nameDict, descriptions_dict = descriptionDict)

if __name__ == '__main__':
	# app.run()
	port = int(os.environ.get('PORT', 5000))
	app.run(host='0.0.0.0', port=port)