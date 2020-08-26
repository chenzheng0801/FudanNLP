from collections import Counter

def preprocess(text, freq=5):
    """
    :param text: 文本数据
    :param freq: 词频阈值
    :return: 
    对文本进行预处理
    """
    text = text.lower()
    text = text.replace('.', '<PERIOD>')
    text = text.replace(',', '<COMMA>')
    text = text.replace('"', '<QUOTATION_MARK>')
    text = text.replace(';', ' <SEMICOLON> ')
    text = text.replace('!', ' <EXCLAMATION_MARK> ')
    text = text.replace('?', ' <QUESTION_MARK> ')
    text = text.replace('(', ' <LEFT_PAREN> ')
    text = text.replace(')', ' <RIGHT_PAREN> ')
    text = text.replace('--', '<HYPHENS> ')
    text = text.replace('?', ' <QUESTION_MARK> ')
    # text = text.replace('\n', ' <NEW_LINE> ')
    text = text.replace(':', ' <COLON> ')
    words = text.split()

    word_counts = Counter(words)
    trimmed_words = [word for word in words if word_counts[word] > freq]
    return trimmed_words


def build_vocab(data_path):
    with open('data/text8') as f:
        text = f.read()
    word_list = preprocess(text)
    vocab = set(word_list)
    vocab_to_int = {w: c for c, w in enumerate(vocab)}
    int_to_vocab = {c: w for c, w in enumerate(vocab)}



a=0

