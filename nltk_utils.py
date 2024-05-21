import nltk #nanural language toolkit
from nltk.stem.porter import PorterStemmer
import numpy as np
stemmer=PorterStemmer()
#Tokenization is a process in natural language processing (NLP) and text processing that involves splitting text into individual units called tokens. These tokens can be words, phrases, symbols, or other meaningful elements. Tokenization is a crucial step in preparing text data for further analysis, such as in machine learning, text mining, and information retrieval.
def tokenize(sentense):
    return nltk.word_tokenize(sentense)

#Stemming is a process in natural language processing (NLP) that reduces words to their base or root form. The primary purpose of stemming is to group together different forms of a word so they can be analyzed as a single item. This is particularly useful in text mining, information retrieval, and indexing.
def stem(word):
    return stemmer.stem(word.lower())

def bag_Of_words(tokenized_sentense, all_words):
     """
    return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
     tokenized_sentense=[stem(w) for w in tokenized_sentense]
     bag=np.zeros(len(all_words), dtype=np.float32)
     for idx, w in enumerate(all_words):
         if w in tokenized_sentense:
             bag[idx]=1.0

     return bag
 
