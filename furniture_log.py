import numpy as np
import matplotlib.pyplot as plt
import skimage
from skimage import io, filters, transform
import os
import json
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import tensorflow as tf
import numpy as np
import re

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


# In[17]:
train = np.array(load_images_from_folder('furniture_train'))

X_train = train[0]
y_train = train[1]
y_train=y_train.astype('int')
y_train=y_train.reshape(y_train.shape[0],1)

X_train_resized = [transform.resize(img,(300,300,3)) for img in X_train]

X_train_flat = np.array([img.flatten() for img in X_train_resized])

test = np.array(load_images_from_folder('furniture_valid'))
X_test = test[0]
y_test = test[1]
y_test = y_test.astype('int')
y_test=y_test.reshape(y_test.shape[0],1)

X_test_resized = [transform.resize(img,(300,300,3)) for img in X_test]
X_test_flat = np.array([img.flatten() for img in X_test_resized])

# Input dimension
in_dim = X_train_flat.shape[1]

# Initialize placeholders
x = tf.placeholder(shape=[None, in_dim], dtype=tf.float32)
y = tf.placeholder(shape=[None, 1], dtype=tf.float32)

# Create variables for logistic regression
A = tf.Variable(tf.random_normal(shape=[in_dim,1]))
b = tf.Variable(tf.random_normal(shape=[1,1]))

# Declare model operations
model_output = tf.add(tf.matmul(x, A), b)

# Declare loss function 
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=y))

# Declare optimizer
my_opt = tf.train.GradientDescentOptimizer(0.00001)
train_step = my_opt.minimize(loss)

# Map model output to binary predictions
prediction = tf.round(tf.sigmoid(model_output))
predictions_correct = tf.cast(tf.equal(prediction, y), tf.float32)
accuracy = tf.reduce_mean(predictions_correct)

with tf.Session() as sess:
    # Initialize variables
    init = tf.global_variables_initializer()
    sess.run(init)
    sess.graph.finalize()  # graph is read-only after this statement

    # Training loop
    loss_vec = []
    train_acc = []
    test_acc = []
    for i in range(2000):
        sess.run(train_step, feed_dict={x: X_train_flat, y: y_train})
        temp_loss = sess.run(loss, feed_dict={x: X_train_flat, y: y_train})
        loss_vec.append(temp_loss)
        temp_acc_train = sess.run(accuracy, feed_dict={x: X_train_flat, y: y_train})
        train_acc.append(temp_acc_train)
        temp_acc_test = sess.run(accuracy, feed_dict={x: X_test_flat, y: y_test})
        test_acc.append(temp_acc_test)
        if (i+1)%100==0:
            print(i+1, ': Loss =', temp_loss, ', Training acc. =', temp_acc_train,', Test acc. = ', temp_acc_test)

# Plot loss over time
plt.figure(1)
plt.semilogy(loss_vec, 'k-')
plt.title('Cross Entropy')
plt.xlabel('Iteration')
plt.ylabel('Cross Entropy')

# Plot train and test accuracy
plt.figure(2)
plt.plot(train_acc, 'k-', label='Training')
plt.plot(test_acc, 'r--', label='Test')
plt.title('Accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

print(test_acc)
