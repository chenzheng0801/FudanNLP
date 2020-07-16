import nltk
from nltk.stem import WordNetLemmatizer
import csv
from collections import Counter
import re

nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
stopwords = ['i', 'me', 'my', 'myself', 'we', 'our', 'of',
             'ours', 'ourselves', 'you', 'your', 'yours',
             'yourself', 'yourselves', 'he', 'him', 'his',
             'himself', 'she', 'her', 'to', 'hers', 'herself',
             'it', 'its', 'itself', 'they', 'them', 'their',
             'theirs', 'themselves', 'what', 'which', 'who',
             'whom', 'this', 'that', 'these', 'those', 'am',
             'is', 'are', 'was', 'were', 'be', 'been', 'being',
             'have', 'has', 'had', 'having', 'do', 'does',
             'did', 'doing', 'a', 'an', 'the', 'and', ',', '.',
             '-s', '-ly', '</s>', 's']


def read_tsv(file_path):
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


def data_cleaning(data_map):
    clean_data = {}
    for key, value in data_map.items():
        phrase, label, ph_len = value
        words = phrase.lower().split()
        meaningful_words = [word for word in words if word not in stopwords]
        lemma_list = list(map(lemmatizer.lemmatize, meaningful_words))
        clean_data[key] = (lemma_list, label)
    return clean_data


def build_vocab(clean_data):
    count_list = []
    no_repeat_word_list = []
    document_count = Counter()
    for _, value in clean_data.items():
        lemma_list, label = value
        counter = Counter(lemma_list)
        count_list.append(counter)
        word_list = list(set(lemma_list))
        no_repeat_word_list.append(word_list)
        for word in word_list:
            document_count[word] += 1
    return no_repeat_word_list, count_list, document_count

if __name__ == "__main__":
    # raw_review = "afsdfas asdfawef asda?as"
    # letters_only = re.sub('[^a-zA-Z]', ' ', raw_review)
    data_map = read_tsv("train.tsv")
    clean_data = data_cleaning(data_map)
    no_repeat_word_list, count_list, document_count = build_vocab
