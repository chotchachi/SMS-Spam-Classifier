from gensim.parsing.preprocessing import strip_non_alphanum, strip_multiple_whitespaces,preprocess_string, split_alphanum, strip_short, strip_numeric
from sklearn.feature_extraction.text import CountVectorizer
import string
import re
import codecs
from unidecode import unidecode
import numpy as np

def load_corpus(filename):
    list_label = []
    list_content = []
    f = codecs.open(filename, 'r', 'utf-8')
    for line in f:
        line = line.strip().split('\t')
        list_label.append(line[0])
        list_content.append(line[1])
        # print('line >>>',line)
    return list_label, list_content


#load stop words from txt
def load_stopwordCorpus(filename):
    list_content = []
    f = codecs.open(filename, 'r', 'utf-8')
    for line in f:
        line = line.strip().split('\t')
        list_content.append(line[0])
    return  list_content

# convert text in curpus to vectors
def convertToVector(corpus,bagwods):
    vectors = []
    for doc in corpus:
        vector = np.zeros(len(bagwods))
        for i, word in enumerate(bagwods):
            if word in doc:
                vector[i] = 1
        vectors.append(vector)
    return vectors

# map labels ham -> 0 , spam -> 1
def mapLabelToNumber(labels):
    output = []
    for index,label in enumerate(labels):
        if label == 'ham':
            output.append(0)
        else:
            output.append(1)
    return output

# load bag_wordf from courpus
def configure_bag_words(corpus):
    set_words = []
    for doc in corpus:
        words = doc.split(' ')
        set_words += words
        set(set_words)
    set_words = list(set(set_words))
    return set_words

# remove special character
def remove_special_character(text):
    output = text.lower()
    removeLine = re.sub(r"http\S+", "", output)
    nopunc = [char for char in removeLine if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    doc = strip_non_alphanum(nopunc).lower().strip() #remove emoji
    # remove cac tu ket hop voi cac so b1,b2..
    doctest = split_alphanum(doc)
    document_test = strip_short(doctest, minsize=2) # xoa cac ky tu dung 1 minh
    remove_number_doc = strip_numeric(document_test) # xoa cac con so
    # rawdoc = ViTokenizer.tokenize(remove_number_doc)
    a = unidecode(remove_number_doc)
    return a# ghép các từ có nghĩa trong tiếng việt


def remove_stopword_from_text(message,stopwords):
    output  = ''
    words = message.split(' ')
    for word in words:
        if (word not in stopwords):
            output += word
    return output

def smoothing(a, b):
    return float((a+1)/(b+1))

#get count spam or nonspam
def getCount_Spam_or_Nonspam(labels):
    spam = 0
    non_spam = 0
    for l in labels:
        if l == 1:
            spam += 1
        else:
            non_spam += 1
    return spam,non_spam

# define bayes matrix
def configure_bayes_matrix(bag_words,vectors,labels,spam,non_spam,bayes_matrix):
    for i, word in enumerate(bag_words):
        app_spam = 0 # số lần spam vectors chứa từ này
        app_nonspam = 0 # số lần ham vectors chứa từ này
        nonapp_spam = 0 # số lần vector spam không chứa từ này
        nonapp_nonspam = 0 # số lần ham vectors không chứa từ này
        # tổng số lần n = length(vector)
        for k, v in enumerate(vectors):
            if v[i] == 1:
                if labels[k] == 1:
                    app_spam += 1
                else:
                    app_nonspam += 1
            else:
                if labels[k] == 1:
                    nonapp_spam += 1
                else:
                    nonapp_nonspam += 1

        bayes_matrix[i][0] = smoothing(app_spam, spam) # p x / spam
        bayes_matrix[i][1] = smoothing(app_nonspam, non_spam) # p x/ham
        bayes_matrix[i][2] = smoothing(nonapp_spam, spam) # p !x/spam
        bayes_matrix[i][3] = smoothing(nonapp_nonspam, non_spam) #p !x/ham

# create vector from message
def handleMessage(message,bag_words):
    input = remove_special_character(message)
    vector = np.zeros(len(bag_words))
    for i, word in enumerate(bag_words):
        if word in input:
            vector[i] = 1
    return vector

#compare rate spam or nonspam
def compare(predict_spam, predict_non_spam):
    rate_spam = predict_spam/(predict_spam+predict_non_spam)
    rate_nonspam = predict_non_spam/(predict_spam+predict_non_spam)
    if rate_spam > rate_nonspam:
        return True
    return False

# predict message
def predict(message,bag_words,_p_spam,_p_nonspam,bayes_matrix):
    vector_message = handleMessage(message,bag_words)

    predict_spam = _p_spam
    predict_non_spam = _p_nonspam

    for i, v in enumerate(vector_message):
        if v == 0:
            predict_spam *= bayes_matrix[i][2]
            predict_non_spam *= bayes_matrix[i][3]
        else:
            predict_spam *= bayes_matrix[i][0]
            predict_non_spam *= bayes_matrix[i][1]
    if compare(predict_spam, predict_non_spam):
        return 1
    return 0

