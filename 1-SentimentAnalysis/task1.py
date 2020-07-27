from sklearn.feature_extraction.text import CountVectorizer
import csv
from nltk.corpus import wordnet
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
import csv
import numpy as np
from sklearn.linear_model import LogisticRegression
from collections import Counter
from sklearn import preprocessing
import math

stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'of', "'s",
                 'ours', 'ourselves', 'you', 'your', 'yours', 'in', 'on',
                 'yourself', 'yourselves', 'he', 'him', 'his', 'by',
                 'himself', 'she', 'her', 'to', 'hers', 'herself',
                 'it', 'its', 'itself', 'they', 'them', 'their', '..',
                 'theirs', 'themselves', 'what', 'which', 'who', '...',
                 'whom', 'this', 'that', 'these', 'those', 'am', '\'',
                 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                 'have', 'has', 'had', 'having', 'do', 'does', 'with',
                 'did', 'doing', 'a', 'an', 'the', 'and', ',', '.',
                 '-s', '-ly', '</s>', 's', ',,', ',,,', "``", "\"",
                 '\'\'', "\"\"", ":"]


def read_tsv(file_path):
    csv.register_dialect("mydialect", delimiter='\t', quoting=csv.QUOTE_ALL)
    data_map = {}
    with open(file_path, ) as csvfile:
        data = csv.reader(csvfile, 'mydialect')
        count = 0
        for line in data:
            if count > 0:
                sent_id, phrase, label = line[1], line[2], line[3]
                ph_len = len(phrase)
                if sent_id not in data_map.keys():
                    data_map[sent_id] = (phrase, label, ph_len)
                else:
                    t_phrase, t_label, t_len = data_map[sent_id]
                    if ph_len > t_len:
                        data_map[sent_id] = (phrase, label, ph_len)
            count += 1
    return data_map

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None


def lemmatize_sentence(sentence):
    res = []
    lemmatizer = WordNetLemmatizer()
    for word, pos in pos_tag(word_tokenize(sentence)):
        wordnet_pos = get_wordnet_pos(pos) or wordnet.NOUN
        res.append(lemmatizer.lemmatize(word, pos=wordnet_pos))

    return res

def data_cleaning(data_map):
    clean_data = {}
    for key, value in data_map.items():
        phrase, label, ph_len = value
        tokens = lemmatize_sentence(phrase.lower())
        meaningful_tokens = [token for token in tokens if token not in stopwords]
        clean_data[key] = (" ".join(meaningful_tokens), label)
    return clean_data


data_map = read_tsv("train.tsv")
clean_data = data_cleaning(data_map)
count_vec = CountVectorizer()

lemma_list = []
y_train = []
for _, value in clean_data.items():
    lemma_phrase, label = value
    lemma_list.append(lemma_phrase)
    y_train.append(label)

x_count_train = count_vec.fit_transform(lemma_list)
logist = LogisticRegression()
logist.fit(x_count_train, y_train)
x_test = x_count_train
predicted = logist.predict(x_test)
print(np.mean(predicted == y_train))