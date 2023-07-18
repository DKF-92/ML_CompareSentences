# This script will be responsible for creating a model based as our csv file

# Imports dependencies

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt') 
import pandas as pd

# Imports data
data=pd.read_csv('ML_Compare Sentences\statements_short.csv')
quest1_list = data['question1'].values.tolist()

# Cleans data: removes nulls, extra characters etc.
tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(quest1_list)]

max_epochs = 100
vec_size = 20
alpha = 0.025

# Vectorizes model
model = Doc2Vec(vector_size=vec_size,
                alpha=alpha,
                min_alpha=0.00025,
                min_count=1,
                dm=1)

model.build_vocab(tagged_data)

for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(tagged_data,
                total_examples = model.corpus_count,
                epochs = 40)
    # decrease learning rate
    model.alpha -= 0.0002
    # fix the learning rate
    model.min_alpha = model.alpha

model.save("compare_d2v.model")
print("Model Saved") 