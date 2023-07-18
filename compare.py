# Compares 2 statements for similarity
from gensim.models.doc2vec import Doc2Vec
from nltk.tokenize import word_tokenize

# Loads recently made model 
model = Doc2Vec.load("ML_Compare Sentences\compare_d2v.model")

# Sets test data 1 - already in data, 2,3 - to test similarity
testData1 = word_tokenize("How can I increase the speed of my internet connection while using a VPN?".lower())
testData2 = word_tokenize("How can i increase speed internet connection?".lower())
testData3 = word_tokenize("Where is London?".lower())

# Prints similarity results
print(model.wv.n_similarity(testData1,testData1))
print(model.wv.n_similarity(testData1,testData2))
print(model.wv.n_similarity(testData1,testData3))