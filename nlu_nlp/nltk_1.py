import nltk
import numpy as np

nltk.download('punkt')  # Download tokenizer if not already downloaded
nltk.download('averaged_perceptron_tagger')  # Download POS tagger if not already downloaded

stemmer = nltk.PorterStemmer()

def tokenize(sentence):
    """
    Split sentence into array of words/tokens
    A token can be a word or punctuation character, or number
    """
    return nltk.word_tokenize(sentence)

def stem(word):
    """
    Stemming = find the root form of the word
    Examples:
    words = ["organize", "organizes", "organizing"]
    words = [stem(w) for w in words]
    -> ["organ", "organ", "organ"]
    """
    return stemmer.stem(word.lower())

def tag_parts_of_speech(sentence):
    if isinstance(sentence, str):
        tokenized_sentence = nltk.word_tokenize(sentence)
        pos_tags = nltk.pos_tag(tokenized_sentence)
    elif isinstance(sentence, list):
        tokenized_sentence = [nltk.word_tokenize(s) for s in sentence]
        pos_tags = [nltk.pos_tag(tokens) for tokens in tokenized_sentence]
    else:
        raise TypeError("sentence: expected a string or a list of strings")

    return tokenized_sentence, pos_tags

def bag_of_words(tokenized_sentence, words):
    """
    Return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    Example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bog   = [  0 ,    1 ,    0 ,    1 ,    0 ,    0 ,      0]
    """
    # Stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    # Initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag


# def get_wordnet_part_of_speech(tag):

#   if tag.startswith("J"):
#     return wordnet.ADJ

#   elif tag.startswith("V"):
#     return wordnet.VERB

#   elif tag.startswith("N"):
#     return wordnet.NOUN

#   elif tag.startswith("R"):
#     return wordnet.ADV

#   else:
#     return wordnet.NOUN

# def lemmatize(x):
#   text = []
#   tokens = word_tokenize(x)
#   words_tags = nltk.pos_tag(tokens)
#   for word, tag in words_tags:
#     num =  WordNetLemmatizer().lemmatize(word, pos=get_wordnet_part_of_speech(tag))
#     text.append(num)
#     return " ".join(text)