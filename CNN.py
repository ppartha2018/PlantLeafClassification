# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from keras.applications.vgg16 import VGG16


# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss
import os
import sys
import pickle
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.utils.vis_utils import plot_model
from keras.utils import np_utils
from sklearn.decomposition import PCA
from sklearn.feature_extraction.image import extract_patches_2d

K.set_image_dim_ordering('tf')
dir_prefix = "../data/all_data/"
data_file_dir = ["leaf1" , "leaf2", "leaf3", "leaf4", "leaf5", "leaf6", "leaf7", "leaf8", "leaf9", "leaf10", "leaf11", "leaf12", "leaf13", "leaf14", "leaf15"]
#classes_dict = {"leaf1" : , "leaf2", "leaf3", "leaf4", "leaf5", "leaf6", "leaf7", "leaf8", "leaf9", "leaf10", "leaf11", "leaf12", "leaf13", "leaf14", "leaf15"}
test_split = 0.35
n_folds = 5 #k-fold cross validation
pca_weights = None


'''
Read images and preprocess the input data.
Precproessing include: cropping, reshaping, normalizing the input RGB values to [0,1] scale.
'''
def read_preprocess_data():
    X_data = []
    X_classes = []
    print("Reading Input Data: ")
    i = 0
    for dir in data_file_dir:
        for file in os.listdir(dir_prefix + dir):
            img = Image.open(dir_prefix + dir + "/" + file)
            width, height = img.size   # Get dimensions
            left = width/10
            top = height/10
            right = 9 * width/10
            bottom = 9 * height/10
            #img = img.crop((left, top, right, bottom))
            img = img.resize((50,50))
            arr = np.array(img)
            #normalizing the pixel values to the scale [0,1]
            arr = arr / 255
            X_data.append(arr)
            #cropping 'leaf1' to int('leaf'1'')
            X_classes.append(i)
            sys.stdout.write("->")
            sys.stdout.flush()
        i = i + 1
    X_data = np.array(X_data)
    X_classes = np.array(X_classes)
    print("len(X_data): ", len(X_data))
    print("X_data.shape: ", X_data.shape)
    train_data, test_data, train_classes, test_classes = train_test_split(X_data, X_classes, test_size = test_split, random_state=42)
    print("Train Data, Test Data, Train Classes, Test Classes:")
    print(type(train_data), type(test_data), type(train_classes), type(test_classes))
    print(len(train_data), len(test_data), len(train_classes), len(test_classes))
    print(train_data.shape, test_data.shape, train_classes.shape, test_classes.shape)
    print("Test Classes: ", test_classes)
    return train_data, test_data, train_classes, test_classes

'''
Creting a multi-layer CNN.
Relu - acivation for Convolution layers
Softmax - acivation for Fully Connected Layer leading for multi-class predictions
'''
def create_model():
  model = Sequential()
  #model.add(Conv2D(25, (5, 5), input_shape=(50, 50, 3)))
  model.add(Conv2D(25, (5, 5), input_shape=(50, 50, 3)))
  #print("**** Model Initialized weights: ", len(model.get_weights()), model.get_weights()[0].shape)
  #print(model.get_weights())
  #model.add(Conv2D(25, (5, 5), input_shape=(50, 50, 3), kernel_initializer=my_init))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))

  #model.add(Conv2D(32, (3, 3)))
  #model.add(Activation('relu'))
  #model.add(MaxPooling2D(pool_size=(2, 2)))

  #model.add(Conv2D(64, (3, 3)))
  #model.add(Activation('relu'))
  #model.add(MaxPooling2D(pool_size=(2, 2)))
  # the model so far outputs 3D feature maps (height, width, features)
  model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
  model.add(Dense(1024))
  model.add(Activation('relu'))
  #model.add(Dropout(0.5))
  model.add(Dense(15))
  model.add(Activation('softmax'))
  plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
  model.compile(loss='categorical_crossentropy',
              optimizer = tf.train.AdamOptimizer(),
              metrics=['categorical_accuracy'])
  model.summary()
  '''
  model = VGG16()
  model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
  '''
  return model
  
def train_evaluate_model(model, train_data, train_classes, validate_data, validate_classes):
  model.fit(train_data, train_classes, epochs=10)
  test_loss, test_acc = model.evaluate(validate_data, validate_classes)
  predictions = model.predict(validate_data)
  print('Test accuracy:', test_acc, "Test loss: ", test_loss)
  print("Validate Classes, Predictions:")
  #print(validate_classes, predictions)
  print("Length of predictions: ", len(predictions), "Length of validate classes: ", len(validate_classes))
  #llog_loss = log_loss(validate_classes, predictions)
  #print("Log Loss: ", llog_loss)


def plot_image(i, predictions_array, true_label, img):
  #predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)   
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'


  ''' 
  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)
  '''

  plt.show()

def save_array(arr , file):
    output = open(file, 'wb')
    pickle.dump(arr, output)
    output.close()

def load_array(file):
    pkl_file = open(file, 'rb')
    data = pickle.load(pkl_file)
    return data

def my_init(shape, dtype=None):
    print("CAlling my initializer: shape: ", shape, "dtype: ", dtype)
    ker = np.zeros(shape, dtype=dtype)
    #ker = np.ones()
    print("Kernel Shape to return: ",pca_weights.shape)
    return pca_weights


# <------------------------ Program Start: -------------------------------------------------------->

print("Tensorflow version: ",tf.__version__)
'''
all_train_data, test_data, all_train_classes, test_classes = read_preprocess_data()
save_array(all_train_data, "all_train_data.pk1")
save_array(all_train_classes, "all_train_classes.pk1")
save_array(test_data, "test_data.pk1")
save_array(test_classes, "test_classes.pk1")
'''
all_train_data = load_array("all_train_data.pk1")
all_train_classes = load_array("all_train_classes.pk1")
all_train_classes = np_utils.to_categorical(all_train_classes)
test_data = load_array("test_data.pk1")
test_classes = load_array("test_classes.pk1")
#print(test_classes)
test_classes = np_utils.to_categorical(test_classes)
#print(test_classes[0])

'''
PCA initialization of weights.

Generate patches from inputs, run PCA, get first first few principal components and use them as weight vectors.
Reference:

https://ieeexplore.ieee.org/abstract/document/6706969

'''

######### Start PCA ##############

print("Running PCA on random patches: ")
print("All train data shape: ", all_train_data.shape)
all_train_data_pca = all_train_data
all_train_data_pca = all_train_data_pca.reshape(-1,2500)
print("After all_train_data reshape: ", all_train_data_pca.shape)
patches = extract_patches_2d(all_train_data_pca,  (5,5), max_patches=75)
print("patches.shape: ", patches.shape)
patches = patches.reshape(-1,75)
print("after patches re-shape: ", patches.shape)
pca = PCA(n_components=25, random_state=0, svd_solver='randomized')
pca.fit(patches)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.ylim(0.8, 1.0)
plt.grid()
plt.show()
print("pca.components_.shape",pca.components_.shape)
print("pca.components_[0].shape",pca.components_[0].shape)
pca_weights = np.reshape(pca.components_, (5,5,3,25))
print("pca_weights.shape: ", pca_weights.shape)
print("End PCA! ")

######## End PCA ################



model = create_model()
train_evaluate_model(model, all_train_data, all_train_classes, test_data, test_classes)

#Now test with the unseen data
#model.fit(train_data, train_classes, epochs=5)
#test_loss, test_acc = model.evaluate(test_data, test_classes)