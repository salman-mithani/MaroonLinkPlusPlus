import json
import requests
import nltk
import string
import re
import html
from collections import Counter
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize

# stop_words = stopwords.words('english') + list(string.punctuation)
stop_words = stopwords.words('english')
ignore = ["nbsp", "amp"]

def strategy2(revs):
    id_tokens = []
    for r in revs:
        rev = r.lower()
        regx = re.compile('\W+')
        tokens = [t for t in regx.split(rev) if len(t)>=3 and t.isalpha() and t not in stop_words and t not in ignore]
        id_tokens.append(tokens)
    return id_tokens

def tokenize(r):
	rev = r.lower()
	regx = re.compile('\W+')
	tokens = [t for t in regx.split(rev) if len(t)>=3 and t.isalpha() and t not in stop_words and t not in ignore]
	return tokens

def ten_most_popular_words(revs):
    all_tokens = []
    for r in revs:
        tokens = tokenize(r)
        all_tokens += tokens
    counts = Counter(all_tokens)
    most_frequent = counts.most_common(10)
    return most_frequent

def cleanhtml(raw_html):
	cleanr = re.compile('<.*?>')
	cleantext = re.sub(cleanr, '', raw_html)
	return cleantext

filename = "all_organization_data.json"

with open(filename) as f:
	data = json.load(f)

keys = []
names = []
descriptions = []
summaries = []

nameDict = {}
descriptionDict = {}
summaryDict = {}

for e in data["value"]:
	keys.append(e["WebsiteKey"])
	names.append(e["Name"])
	nameDict[e["WebsiteKey"]] = e["Name"]
	if e["Description"]:
		descriptions.append(cleanhtml(html.unescape(e["Description"])))
		descriptionDict[e["WebsiteKey"]] = cleanhtml(html.unescape(e["Description"]))
	else:
		descriptions.append(e["Name"])
		descriptionDict[e["WebsiteKey"]] = e["Name"]
	if e["Summary"]:
		summaries.append(cleanhtml(html.unescape(e["Summary"])))
		summaryDict[e["WebsiteKey"]] = cleanhtml(html.unescape(e["Summary"]))
	else:
		summaries.append(e["Name"])
		summaryDict[e["WebsiteKey"]] = e["Name"]

# print(len(descriptions))
# print(nameDict)
# print(descriptionDict["studentactivities"])
# print(summaryDict["studentactivities"])
# print(len(summaryDict))
# print(summaryDict["30loves"])

# out_name = "orgKeys.txt"
# f = open(out_name, "w")
# f.write("\n".join(keys))

# out_name = "orgDescriptions.json"
# f = open(out_name, "w")
# json.dump(descriptionDict, f)

# out_name = "orgNames.json"
# f = open(out_name, "w")
# json.dump(nameDict, f)

# out_name = "orgSummaries.json"
# f = open(out_name, "w")
# json.dump(summaryDict, f)


# print("Number of organizations:", len(keys))

# r = requests.get("https://maroonlink.tamu.edu/api/discovery/organization/bykey/Aco")

# print(r.json()["description"])



# tokens = strategy2(descriptions)
# print(tokens[1])

# s = " ".join(descriptions)
# l = s.split(" ")
# counts = Counter(l)
# print(counts.most_common(10))

# word_tokens = word_tokenize(s)
# filtered_sentence = [w for w in word_tokens if not w in stop_words] 

# print(filtered_sentence)
# counts = Counter(filtered_sentence)
# print(counts.most_common(10))

# m = ten_most_popular_words(descriptions)
# print(m)



