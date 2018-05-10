import os
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import re
from skimage import transform
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MultiLabelBinarizer

seed = 128
rng = np.random.RandomState(seed)

mlb=MultiLabelBinarizer()

# # read data
def load_images_from_folder(folder):
    X = []
    y = []
    for filename in os.listdir(folder):
        img = io.imread(os.path.join(folder,filename))
        if img is not None:
            label = re.findall(r'_([^.]*).',filename)
            X.append(img)
            y.append(int(label[0]))
    return X,y

train = np.array(load_images_from_folder('bigger_train'))

X_train = train[0]
y_train = train[1]
y_train=y_train.astype('int')

X_train_resized = [transform.resize(img,(50,50,3)) for img in X_train]

X_train_flat = np.array([img.flatten() for img in X_train_resized])

test = np.array(load_images_from_folder('furniture_valid'))
X_test = test[0]
y_test = test[1]
y_test = y_test.astype('int')


X_test_resized = [transform.resize(img,(50,50,3)) for img in X_test]
X_test_flat = np.array([img.flatten() for img in X_test_resized])

ipca=IncrementalPCA(n_components=300, batch_size=None,whiten=True)
ipca.fit(X_train_flat)
#X_train_flat=ipca.transform(X_train_flat)
#X_test_flat=ipca.transform(X_test_flat)

x_train=X_train_flat
x_test=X_test_flat
y_train=y_train-1
y_test=y_test-1

all_labels=[]
for i in range(1,129):
    all_labels.append([i])
mlb.fit(all_labels)
#y_train=mlb.transform(y_train)
#y_test=mlb.transform(y_test)

# Initialize placeholders 
x = tf.placeholder(dtype = tf.float32, shape = [None, 300])
y = tf.placeholder(dtype = tf.int32, shape = [None])

W1 = tf.Variable(tf.random_normal([300, 300], stddev=0.03), name='W1')
b1 = tf.Variable(tf.random_normal([300]), name='b1')
# and the weights connecting the hidden layer to the output layer
W2 = tf.Variable(tf.random_normal([300, 128], stddev=0.03), name='W2')
b2 = tf.Variable(tf.random_normal([128]), name='b2')

# calculate the output of the hidden layer
hidden_out = tf.add(tf.matmul(x, W1), b1)
hidden_out = tf.nn.relu(hidden_out)

logits = tf.nn.softmax(tf.add(tf.matmul(hidden_out, W2), b2))

# maybe add some clipped shit here

# Define a loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, 
                                                                    logits = logits))
# Define an optimizer 
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

# Convert logits to label indexes
correct_pred = tf.argmax(logits, 1)

# Define an accuracy metric
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

tf.set_random_seed(1234)
sess = tf.Session()

sess.run(tf.global_variables_initializer())

for i in range(2001):
        #print('EPOCH', i)
        _, accuracy_val = sess.run([train_op, accuracy], feed_dict={x: x_train, y: y_train})
        if i % 10 == 0:
            print("Loss: ", loss)
        #print('DONE WITH EPOCH')
        print(accuracy_val)

predicted = sess.run([correct_pred], feed_dict={x: x_test})[0]
match_count = sum([int(y == y_) for y, y_ in zip(y_test, predicted)])
# Calculate the accuracy
accuracy = match_count / len(y_test)

# Print the accuracy
print("Accuracy: {:.3f}".format(accuracy))
