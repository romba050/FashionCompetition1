import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage import io, filters, transform
import os
import json
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from skmultilearn.adapt import MLkNN
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import tensorflow as tf

mlb=MultiLabelBinarizer()

os.chdir('imaterial_train')
X_train = np.array([io.imread(str(i)+'.jpg') for i in range(1,201)])
os.chdir('..')

train_json = json.load(open('train.json'))

#filenames=["./imaterial_train/"+str(i)+".jpg" for i in range(1,201)]
#print(filenames)

y_train = []
for i in range(1,201):
    labels = train_json['annotations'][i]['labelId']
    labels = np.array(list(map(int,labels)))
    y_train.append(labels)
y_train = np.array(y_train)
all_labels=[]
for i in range(1,229):
    all_labels.append([i])
mlb.fit(all_labels) # fitting multilabelbinarizer to all labels
y_train=mlb.transform(y_train)


#print('smallest vertical:',min(i.shape[0] for i in X_train))
#print('smallest horizontal:',min(i.shape[1] for i in X_train))

#All images resized to the smallest dimensions
X_train_resized = [transform.resize(img,(200,128,3)) for img in X_train]
X_train_flat = np.array([img.flatten() for img in X_train_resized])

os.chdir('./imaterial_validation')
X_test = np.array([io.imread(str(i)+'.jpg') for i in range(1,201)])
os.chdir('..')

X_test_resized = [transform.resize(img,(200,128,3)) for img in X_test]
X_test_flat = np.array([img.flatten() for img in X_test_resized])

val_json = json.load(open('validation.json'))

y_test = []
for i in range(1,201):
    labels = val_json['annotations'][i]['labelId']
    labels = np.array(list(map(int,labels)))
    y_test.append(labels)
y_test = np.array(y_test)
y_test=mlb.transform(y_test)

knn=KNeighborsClassifier()
knn.fit(X_train_flat,y_train)
pred = knn.predict(X_test_flat)
f1=f1_score(y_test, pred, average='micro')
print("f1 ",f1)

# calc avg f1? or pooled

tru_te=mlb.inverse_transform(pred)
tru_tr=mlb.inverse_transform(y_train)

print(pred)
