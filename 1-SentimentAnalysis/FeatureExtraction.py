from nltk.corpus import wordnet
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
import csv
import numpy as np
from collections import Counter
from sklearn import preprocessing
import math
import re


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


class DataProcess:

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

    def __init__(self, file_path):
        data_map = self.read_tsv(file_path)
        clean_data = self.data_cleaning(data_map)

        no_repeat_word_docs, counter_list, document_count,\
        self.feature_encoder, self.label_one_hot, self.feature_dim = self.build_vocab(clean_data)

        doc_len = len(no_repeat_word_docs)
        self.tf_idf_vecs = []
        self.doc_indicies = []
        for i in range(doc_len):
            counter = counter_list[i]
            doc = no_repeat_word_docs[i]
            self.doc_indicies.append(self.feature_encoder.transform(doc))
            count_sum = sum(counter.values())
            tf_idf_vec = []
            for word in doc:
                tf = counter[word] / count_sum
                idf = math.log(doc_len / document_count[word])
                tf_idf_vec.append(tf * idf)
            self.tf_idf_vecs.append(tf_idf_vec)
        self.doc_indicies = np.array(self.doc_indicies)
        self.tf_idf_vecs = np.array(self.tf_idf_vecs)

    def batch_iter(self, batch_size=32):
        feature_dim = self.feature_dim
        batch_data = np.zeros([batch_size, feature_dim])

        doc_len = len(self.doc_indicies)
        indices = list(np.random.permutation(np.arange(doc_len)))
        doc_indicies = self.doc_indicies[indices]
        one_hot = self.label_one_hot[indices]
        tf_idf_vecs = self.tf_idf_vecs[indices]

        start_id = 0
        while start_id < doc_len:
            end_id = min(start_id + batch_size, doc_len)
            for i in range(start_id, end_id):
                idx = i - start_id
                batch_data[idx, list(doc_indicies[i])] = tf_idf_vecs[i]
            # yield batch_data, one_hot[start_id:end_id]
            start_id += batch_size

    def read_tsv(self, file_path):
        csv.register_dialect("mydialect", delimiter='\t', quoting=csv.QUOTE_ALL)
        data_map = {}
        with open(file_path,) as csvfile:
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

    def data_cleaning(self, data_map):
        clean_data = {}
        for key, value in data_map.items():
            phrase, label, ph_len = value
            tokens = lemmatize_sentence(phrase.lower())
            meaningful_tokens = [token for token in tokens if token not in self.stopwords]
            clean_data[key] = (meaningful_tokens, label)
        return clean_data

    def build_vocab(self, clean_data):
        counter_list = []
        no_repeat_word_docs = []
        document_count = Counter()
        feature_words = set([])

        label_encoder = preprocessing.LabelEncoder()
        one_hot_encoder = preprocessing.OneHotEncoder()
        label_list = []

        for _, value in clean_data.items():
            lemma_list, label = value
            label_list.append([label])
            counter = Counter(lemma_list)
            counter_list.append(counter)
            doc = list(set(lemma_list))
            no_repeat_word_docs.append(doc)
            for word in doc:
                document_count[word] += 1
                feature_words.add(word)
        feature_words = list(feature_words)
        feature_encoder = label_encoder.fit(feature_words)
        return no_repeat_word_docs, counter_list, document_count,\
               feature_encoder, one_hot_encoder.fit_transform(label_list), len(feature_words)


if __name__ == "__main__":
    # raw_review = "afsdfas asdfawef asda?as"
    # letters_only = re.sub('[^a-zA-Z]', ' ', raw_review)
    data_process = DataProcess("train.tsv")
    data_process.batch_iter()