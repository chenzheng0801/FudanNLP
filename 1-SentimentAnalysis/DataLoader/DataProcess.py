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
import pandas as pd

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


def clean_data(file_path):
    df = pd.read_csv(file_path, header=0, delimiter='\t')
    sentiment_series = df['Sentiment']
    phrase_series = df['Phrase']
    sent_id_series = df['SentenceId']

    data_map = {}
    for i, phrase in phrase_series.items():
        sentiment = sentiment_series.at[i]
        sent_id = sent_id_series.at[i]

        ph_len = len(phrase)
        if sent_id not in data_map.keys():
            data_map[sent_id] = (phrase, sentiment, ph_len)
        else:
            _, _, t_len = data_map[sent_id]
            if ph_len > t_len:
                data_map[sent_id] = (phrase, sentiment, ph_len)
    text_list = []
    label_list = []
    for key, value in data_map.items():
        phrase, label, ph_len = value
        tokens = lemmatize_sentence(phrase.lower())
        meaningful_tokens = [token for token in tokens if token not in stopwords]
        text_list.append(" ".join(meaningful_tokens))
        label_list.append([label])
    return text_list, label_list


class FeatureExtraction:
    def __init__(self, raw_docs, labels):
        self.cont_vec = CountVectorizer()
        self.cont_vec.fit(raw_docs)
        self.one_hot_encoder = preprocessing.OneHotEncoder()
        self.one_hot_encoder.fit(labels)

    def fit_y(self, label_list):
        return self.one_hot_encoder.transform(label_list).toarray()

    def fit_x(self, text_list):
        return self.cont_vec.transform(text_list).toarray()

    def batch_iter(self, label_list, text_list, batch_size):
        X_train = self.cont_vec.transform(text_list)
        Y_train = self.one_hot_encoder.transform(label_list)
        txt_len = len(text_list)
        indices = list(np.random.permutation(np.arange(txt_len)))

        X_train = X_train[indices]
        Y_train = Y_train[indices]
        start_id = 0
        while start_id < txt_len:
            end_id = min(start_id + batch_size, txt_len)
            batch_X = X_train[start_id: end_id].toarray()
            batch_Y = Y_train[start_id: end_id].toarray()
            yield batch_X, batch_Y
            start_id += batch_size


# text_list, label_list = clean_data("D:\\NLP_coding\\FudanNLP\\1-SentimentAnalysis\\train.tsv")
# hehe = FeatureExtraction(text_list,[[1],[2],[3],[4],[5]])
# hehe.batch_iter(label_list, text_list, batch_size=32)
# print("ffa")
