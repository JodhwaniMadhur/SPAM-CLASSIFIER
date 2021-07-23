import os
import tarfile
import urllib.request
import email
import email.policy
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import re
from html import unescape
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier


DOWNLOAD_ROOT = "http://spamassassin.apache.org/old/publiccorpus/"
HAM_URL = DOWNLOAD_ROOT + "20030228_easy_ham.tar.bz2"
SPAM_URL = DOWNLOAD_ROOT + "20030228_spam.tar.bz2"
SPAM_PATH = os.path.join("datasets", "spam")

def fetch_spam_data(spam_url=SPAM_URL, spam_path=SPAM_PATH):
    if not os.path.isdir(spam_path):
        os.makedirs(spam_path)
    for filename, url in (("ham.tar.bz2", HAM_URL), ("spam.tar.bz2", spam_url)):
        path = os.path.join(spam_path, filename)
        if not os.path.isfile(path):
            urllib.request.urlretrieve(url, path)
        tar_bz2_file = tarfile.open(path)
        tar_bz2_file.extractall(path=SPAM_PATH)
        tar_bz2_file.close()
         
fetch_spam_data()#calling function to download zip files and extracting them to make datasets




HAM_DIR = os.path.join(SPAM_PATH, "easy_ham")#segregating ham mails into ham folder and spam into spam folder
SPAM_DIR = os.path.join(SPAM_PATH, "spam")
ham_filenames = [name for name in sorted(os.listdir(HAM_DIR)) if len(name) > 20]#segregating on basis of length of name of email file
spam_filenames = [name for name in sorted(os.listdir(SPAM_DIR)) if len(name) > 20]

#print(len(ham_filenames),len(spam_filenames))

#

def load_email(is_spam, filename, spam_path=SPAM_PATH):
    directory = "spam" if is_spam else "easy_ham"
    with open(os.path.join(spam_path, directory, filename), "rb") as f:
        return email.parser.BytesParser(policy=email.policy.default).parse(f)

ham_emails = [load_email(is_spam=False, filename=name) for name in ham_filenames]#dataset of ham emails
spam_emails = [load_email(is_spam=True, filename=name) for name in spam_filenames]#dataset of spam emails
#print(ham_emails[1].get_content().strip())
#print(spam_emails[6].get_content().strip())



def get_email_structure(email):
    if isinstance(email, str):
        return email
    payload = email.get_payload()
    if isinstance(payload, list):
        return "multipart({})".format(", ".join([
            get_email_structure(sub_email)
            for sub_email in payload
        ]))
    else:
        return email.get_content_type()
        #returns type of text(incase it is HTML or contains attachments,which is in outr case)



def structures_counter(emails):
    structures = Counter()
    #Dict subclass for counting hashable items. Sometimes called a bag or multiset. 
    #Elements are stored as dictionary keys and their counts are stored as dictionary values.
    for email in emails:
        structure = get_email_structure(email)
        structures[structure] += 1
    return structures
#the above function returns the number of different structures it found in all emails and their respective counts
#It is like a dictionary.
#print(structures_counter(ham_emails).most_common())

#print(structures_counter(spam_emails).most_common())

#find the one with highest value in spam email and return it's key.


#for header, value in spam_emails[0].items():
    #print(header,":",value)


#print(spam_emails[0]["Subject"])

X = np.array(ham_emails + spam_emails, dtype=object)
y = np.array([0] * len(ham_emails) + [1] * len(spam_emails))
#converting data for splitting it

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#splitting data




def html_to_plain_text(html):
    text = re.sub('<head.*?>.*?</head>', '', html, flags=re.M | re.S | re.I)
    text = re.sub('<a\s.*?>', ' HYPERLINK ', text, flags=re.M | re.S | re.I)
    text = re.sub('<.*?>', '', text, flags=re.M | re.S)
    text = re.sub(r'(\s*\n)+', '\n', text, flags=re.M | re.S)
    return unescape(text)
#the above function converts the html into plain text; PS:this could be done by Beautiful Soup Function.

html_spam_emails = [email for email in X_train[y_train==1]
                    if get_email_structure(email) == "text/html"]
sample_html_spam = html_spam_emails[7]
#print(sample_html_spam.get_content().strip()[:1000], "...")


#print(html_to_plain_text(sample_html_spam.get_content())[:1000], "...")

def email_to_text(email):
    html = None
    for part in email.walk():
        ctype = part.get_content_type()
        if not ctype in ("text/plain", "text/html"):
            continue
        try:
            content = part.get_content()
        except: # in case of encoding issues
            content = str(part.get_payload())
        if ctype == "text/plain":
            return content
        else:
            html = content
    if html:
        return html_to_plain_text(html)

#print(email_to_text(sample_html_spam)[:100], "...")


try:
    import nltk #Natural Language Toolkit


    stemmer = nltk.PorterStemmer()#an algorithm for suffix stripping
    for word in ("Computations", "Computation", "Computing", "Computed", "Compute", "Compulsive"):
        print(word, "=>", stemmer.stem(word))
except ImportError:
    #print("Error: stemming requires the NLTK module.")
    stemmer = None

try:
    import urlextract # may require an Internet connection to download root domain names
    
    url_extractor = urlextract.URLExtract()#for finding and extracting url strings in text
    #print(url_extractor.find_urls("Will it detect github.com and https://youtu.be/7Pq-S557XQU?t=3m32s"))
except ImportError:
    #print("Error: replacing URLs requires the urlextract module.")
    url_extractor = None


class EmailToWordCounterTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, strip_headers=True, lower_case=True, remove_punctuation=True,
                 replace_urls=True, replace_numbers=True, stemming=True):
        self.strip_headers = strip_headers
        self.lower_case = lower_case
        self.remove_punctuation = remove_punctuation
        self.replace_urls = replace_urls
        self.replace_numbers = replace_numbers
        self.stemming = stemming
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X_transformed = []#list 
        for email in X:
            text = email_to_text(email) or ""
            if self.lower_case:
                text = text.lower()#conversion of text to lower case
            if self.replace_urls and url_extractor is not None:
                urls = list(set(url_extractor.find_urls(text)))#list of all urls found from text 
                urls.sort(key=lambda url: len(url), reverse=True)#sort all urls according to length
                for url in urls:
                    text = text.replace(url, " URL ")#replacing all urls from the text with the word URL
            if self.replace_numbers:
                text = re.sub(r'\d+(?:\.\d*)?(?:[eE][+-]?\d+)?', 'NUMBER', text)
                #match pattern in text for any number and replace it with string "NUMBER"
            if self.remove_punctuation:
                text = re.sub(r'\W+', ' ', text, flags=re.M)#removing all pnctuations in text
            word_counts = Counter(text.split())#counting all words
            if self.stemming and stemmer is not None:
                stemmed_word_counts = Counter()
                for word, count in word_counts.items():
                    stemmed_word = stemmer.stem(word)
                    stemmed_word_counts[stemmed_word] += count#dictionary 
                word_counts = stemmed_word_counts
            X_transformed.append(word_counts)#list of dictionary
        return np.array(X_transformed)

X_few = X_train[:3]#just for demo purpose we have transformed first 3 emails here
X_few_wordcounts = EmailToWordCounterTransformer().fit_transform(X_few)
#print(X_few_wordcounts)


from scipy.sparse import csr_matrix

class WordCounterToVectorTransformer(BaseEstimator, TransformerMixin):
    #converting list of dictionaries into Vectors
    def __init__(self, vocabulary_size=1000):
        self.vocabulary_size = vocabulary_size
    def fit(self, X, y=None):
        total_count = Counter()
        for word_count in X:
            for word, count in word_count.items():
                total_count[word] += min(count, 10)
        most_common = total_count.most_common()[:self.vocabulary_size]
        self.vocabulary_ = {word: index + 1 for index, (word, count) in enumerate(most_common)}
        return self
    def transform(self, X, y=None):
        rows = []
        cols = []
        data = []
        for row, word_count in enumerate(X):
            for word, count in word_count.items():
                rows.append(row)
                cols.append(self.vocabulary_.get(word, 0))
                data.append(count)
        return csr_matrix((data, (rows, cols)), shape=(len(X), self.vocabulary_size + 1))

vocab_transformer = WordCounterToVectorTransformer(vocabulary_size=10)
X_few_vectors = vocab_transformer.fit_transform(X_few_wordcounts)
#print(X_few_vectors)


#print(X_few_vectors.toarray())

print(vocab_transformer.vocabulary_)

from sklearn.pipeline import Pipeline
#creating a custom pipeline for the same and passing them through this pipeline 
preprocess_pipeline = Pipeline([
    ("email_to_wordcount", EmailToWordCounterTransformer()),
    ("wordcount_to_vector", WordCounterToVectorTransformer()),
])

X_train_transformed = preprocess_pipeline.fit_transform(X_train)

#using logistic regression,since the answer is in binary format and not linear
log_clf = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
score = cross_val_score(log_clf, X_train_transformed, y_train, cv=3, verbose=3)
print(score.mean())




X_test_transformed = preprocess_pipeline.transform(X_test)#passing testing data through pipeline

log_clf = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
log_clf.fit(X_train_transformed, y_train)#teaching the regresso about the data

y_pred = log_clf.predict(X_test_transformed)#predicting pipelined test data
print(50*"*")
print("Precision: {:.2f}%".format(100 * precision_score(y_test, y_pred)))
print("Recall: {:.2f}%".format(100 * recall_score(y_test, y_pred)))


print(50*"*")
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_transformed, y_train)
y_pred=knn.predict(X_test_transformed)


print("KNN results")
print("Precision: {:.2f}%".format(100 * precision_score(y_test, y_pred)))
print("Recall: {:.2f}%".format(100 * recall_score(y_test, y_pred)))

print(50*"*")
dtc=DecisionTreeClassifier()
dtc.fit(X_train_transformed, y_train)
y_pred=dtc.predict(X_test_transformed)
print("DTC results")
print("Precision: {:.2f}%".format(100 * precision_score(y_test, y_pred)))
print("Recall: {:.2f}%".format(100 * recall_score(y_test, y_pred)))
print(50*"*")