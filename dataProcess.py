from PIL import Image as IM
import numpy as np
import os
from feature import NPDFeature
import pickle

facedir = os.listdir("./datasets/original/face/")
nonfacedir = os.listdir("./datasets/original/nonface/")
face_list = []
nonface_list = []
for i in facedir:
    image = IM.open("./datasets/original/face/" + i)
    image = image.resize((24, 24), IM.ANTIALIAS)
    image = image.convert('L')
    face_list.append(image)
for i in nonfacedir:
    image = IM.open("./datasets/original/nonface/" + i)
    image = image.resize((24, 24), IM.ANTIALIAS)
    image = image.convert('L')
    nonface_list.append(image)
dataset = []
for i in face_list:
    F = NPDFeature(np.array(i))
    dataset.append(F.extract())
for i in nonface_list:
    F = NPDFeature(np.array(i))
    dataset.append(F.extract())
y = np.ones((1000, 1))
y[500:999] = -1
x_set = open('dataset.pkl', 'wb')
y_set = open('y.pkl', 'wb')
pickle.dump(dataset, x_set)
pickle.dump(y, y_set)
y_set.close()
x_set.close()
