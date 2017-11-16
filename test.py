import tensorflow
import bson
import numpy as np # linear algebra
import pandas as pd
import io
import matplotlib.pyplot as plt
from skimage.data import imread
import multiprocessing as mp

bson_file = open('/mldata/train_example.bson','rb')
data = bson.decode_file_iter(bson_file)

prod_to_category = dict()

for c, d in enumerate(data):
	product_id = d['_id']
	category_id = d['category_id'] # This won't be in Test data
	prod_to_category[product_id] = category_id
	for e, pic in enumerate(d['imgs']):
		picture = imread(io.BytesIO(pic['picture']))

prod_to_category = pd.DataFrame.from_dict(prod_to_category, orient='index')
prod_to_category.index.name = '_id'
prod_to_category.rename(columns={0: 'category_id'}, inplace=True)

prod_to_category.head()
plt.imshow(picture)
plt.show()
