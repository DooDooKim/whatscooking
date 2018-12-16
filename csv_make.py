import pandas as pd
import numpy as np
import csv
from pandas import DataFrame, Series
from gensim.models import Word2Vec
from gensim.models import FastText

#parameters
v_num = 300 # the number of vectors
iter = 50 # the number of iterations
sg = 1 # 0:cbow 1:skip-gram

# path : train path / input_path : path to save csv file
path = 'C:/Users/10User/Desktop/JH/all/train.json'
input_path = 'C:/Users/10User/Desktop/JH/all/ft300_sg.csv'

#read train data and make ingredients data frame using pandas
data = pd.read_json(path, encoding="utf-8")
ingred = pd.DataFrame(data['ingredients'])

#make array for projection
buf = np.zeros([v_num, 1], np.float32)
vec = np.zeros([v_num,1], np.float32)
idx = 0

# word embedding
print("making word embedding model")
#model = Word2Vec(sentences=ingred['ingredients'], size=v_num, window=5, min_count=1, iter=iter, workers=8, sg=sg)
model = FastText(sentences=ingred['ingredients'], size=v_num, window=5, min_count=1, iter=iter ,workers=8, sg=sg)
print("embedding is done",'\n')

# write csv file
with open(input_path, 'a') as outcsv:
    writer = csv.writer(outcsv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
    print("writing csv file on your computer")
    for ro in range(len(ingred)):
        for co in range(len(ingred['ingredients'][idx])):
            vec = vec + model.wv.get_vector(ingred['ingredients'][ro][co]).reshape(v_num,1)
        buf = vec.reshape(-1, )
        buf = buf /len(ingred['ingredients'][idx])
        writer.writerow(buf)
        vec = np.zeros([v_num, 1], np.float32)
        buf = np.zeros([v_num, 1], np.float32)
        idx = idx + 1
    print("done")