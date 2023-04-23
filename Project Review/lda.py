import os
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')
nltk.download('punkt')

stop_words = stopwords.words('english')

# define the number of topics and the number of iterations
num_topics = 3
num_iters = 100

# create a dictionary to store all the words in the corpus
word_dict = {}

# read all the documents in the 'Dataset' folder
doc_list = []
topic_list = []
for filename in os.listdir('Dataset'):
    with open('Dataset/' + filename,'r',encoding='ISO-8859-1') as f:
        text = f.read().lower()
        doc_list.append(text)
        topic_list.append(filename[:filename.find('.')])

num_topics = len(doc_list)

# create a list of all the unique words in the corpus
temp_set = set([])
for doc_text in doc_list:
    doc_text = re.sub('[^a-zA-Z]',' ', doc_text).lower().split()
    temp_set = temp_set.union(set([token for token in doc_text if token not in stop_words]))
word_list = list(temp_set)

# create a dictionary to map each word to a unique integer index
for i, word in enumerate(word_list):
    word_dict[word] = i

# create a numpy array to store the document-term matrix
doc_term_matrix = np.zeros((len(doc_list), len(word_dict)))

# populate the document-term matrix with the frequency of each word in each document
for i, doc in enumerate(doc_list):
    doc = re.sub('[^a-zA-Z]',' ', doc).lower().split()
    doc = [token for token in doc if token not in stop_words]
    for word in doc:
        doc_term_matrix[i][word_dict[word]] += 1

# randomly initialize the topic-term matrix
topic_term_matrix = np.random.rand(num_topics, len(word_dict))

# normalize the topic-term matrix
for i in range(num_topics):
    topic_term_matrix[i] = doc_term_matrix[i] / np.sum(doc_term_matrix[i])

# perform Gibbs sampling
for i in range(num_iters):
    print('Iteration: ', i+1, '/', num_iters)
    for k in range(len(word_dict)):
        # calculate the probabilities for each topic
        topic_probs = np.zeros(num_topics)
        for t in range(num_topics):
            topic_probs[t] = 1e-308 + topic_term_matrix[t][k]
        topic_probs /= (np.sum(topic_probs))
        # sample a new topic for the current word
        new_topic = np.random.choice(num_topics, p=topic_probs)
        # update the topic-term matrix
        topic_term_matrix[new_topic][k] += 1
        # normalize the topic-term matrix
        topic_term_matrix[new_topic] /= np.sum(topic_term_matrix[new_topic])

# search for a given sentence/phrase's topic
def search_topic(sentence):
    sentence = re.sub('[^a-zA-Z]',' ', sentence).lower().split()
    sentence_words = [token for token in sentence if token not in stop_words]
    topic_probs = np.zeros(num_topics)
    for word in sentence_words:
        if word in word_dict:
            word_index = word_dict[word]
            for t in range(num_topics):
                topic_probs[t] += 1e-308 + topic_term_matrix[t][word_index]
    topic_probs /= np.sum(topic_probs)
    #print(topic_probs)
    return topic_probs.argmax()

input_list = [
    'How do maxwell\'s equations of electromagnetism help in proving the dual nature of matter',
    'Any thermal engine requires a negative temperature gradiant between a heat source and a heat sink in order to work',
    'The various laws that define the behaviour of ideal gases on a large scale are Charles\'s law, Boyle\'s law, Gay Lussac\'s law and Avogadro\'s law, all of which can be combined to form the ideal gas equation',
    'All reactants require a minimum amount of energy for them to react with each other, called activation energy',
    'The tendancy of a cell\'s internal structure and chemicals to oppose the flow of current through it is called its internal resistance'
]

for input_text in input_list:
    topic_index = search_topic(input_text)
    print('Input sentence:', input_text)
    print('Topic:', topic_list[topic_index])