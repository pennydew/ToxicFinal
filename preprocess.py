import string

from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize, TweetTokenizer

lem = WordNetLemmatizer()
tokenizer = TweetTokenizer()


def word_to_pos(tag):
    if tag.startswith("R"):
        return wordnet.ADV
    elif tag.startswith("N"):
        return wordnet.NOUN
    elif tag.startswith("V"):
        return wordnet.VERB
    elif tag.startswith("J"):
        return wordnet.ADJ
    else:
        return None


def lemmatize(sentence):
    res = []
    lemmatizer = WordNetLemmatizer()
    for word, pos in pos_tag(word_tokenize(sentence)):
        wordnet_pos = word_to_pos(pos) or wordnet.NOUN
        res.append(lemmatizer.lemmatize(word, pos=wordnet_pos))
    return " ".join(res)

def clean_text(sentence):
    # remove punctuation
    sentence = sentence.translate(string.punctuation)

    # lower and remove stop words
    sentence = sentence.lower().split()
    stops = set(stopwords.words("english"))
    sentence = [w for w in sentence if not w in stops]
    sentence = " ".join(sentence)

    # stem
    sentence = sentence.split()
    stemmer = SnowballStemmer("english")
    sentence = " ".join([stemmer.stem(word) for word in sentence])

    return sentence


