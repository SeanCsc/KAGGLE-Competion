import os, cv2, random
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns
%matplotlib inline 
from keras import layers
from keras.models import Sequential
from keras.layers import Input, Dropout, Flatten, Conv2D, MaxPooling2D, Dense, Activation
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.utils import np_utils

TRAIN_DIR = '../input/train/'
TEST_DIR = '../input/test/'

ROWS = 64
COLUMNS = 64
CHANNELS = 3

train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)] # use this for full dataset
train_dogs =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'dog' in i]
train_cats =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'cat' in i]

test_images =  [TEST_DIR+i for i in os.listdir(TEST_DIR)]


train_images = train_dogs[:1000] + train_cats[:1000]
random.shuffle(train_images)
test_images =  test_images[:]
def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR) #cv2.IMREAD_GRAYSCALE
    return cv2.resize(img, (ROWS, COLUMNS))

def prep_data(images):
    count = len(images)
    data = np.ndarray((count, ROWS, COLUMNS, CHANNELS), dtype=np.uint8)

    for i, image_file in enumerate(images):
        image = read_image(image_file)
        data[i] = image
       # if i%250 == 0: print('Processed {} of {}'.format(i, count))
    
    return data

train_data = prep_data(train_images)
test_data = prep_data(test_images)

Optimizer = RMSprop(lr = 1e-4)
objective = 'binary_crossentropy'
def model_dogcat():
    model = Sequential()
    model.add(layers.Conv2D(32,(3,3),activation = 'relu', input_shape = (ROWS,COLUMNS,3)))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(64,(3,3),activation = 'relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(128,(3,3),activation = 'relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Conv2D(128,(3,3),activation = 'relu'))
    model.add(layers.MaxPooling2D((2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512,activation = 'relu'))
    model.add(layers.Dense(1,activation = 'sigmoid'))
    model.compile(loss=objective, optimizer=Optimizer, metrics=['accuracy'])
    return model

model =model_dogcat()
nb_epoch = 10
batch_size = 16
class LossHistory(Callback):
    def on_train_begin(self,logs = {}):
        self.losses = []
        self.val_losses = []
    def on_epoch_end(self, batch,logs = {}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto') 

def run_catdog():
    
    history = LossHistory()
    model.fit(train_data, labels, batch_size=batch_size, epochs = nb_epoch,
              validation_split=0.25, verbose=0, shuffle=True, callbacks=[history, early_stopping])
    

    predictions = model.predict_classes(test_data, verbose=0)
    return predictions, history

predictions, history = run_catdog()
#submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
    #                     "Label": predictions})
#submissions.to_csv("DR.csv", index=False, header=True)
predictions = predictions.reshape(np.product(predictions.shape))
predictions.shape
submissions=pd.DataFrame({"id": list(range(1,len(predictions)+1)),
                         "label": predictions})
submissions.to_csv("DR.csv", index=False, header=True)