# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('../input/train.csv') #get a data frame
test_data = pd.read_csv('../input/test.csv')

#train_data.shape
train = (train_data.iloc[:,1:].values).astype('float32')
train_label = (train_data.iloc[:,0].values).astype('float32')
#X_test = (test_data.iloc[:,1:].values).astype('float32')
test = test_data.values.astype('float32')
from keras.utils.np_utils import to_categorical
train_label = to_categorical(train_label)
#train = X_train.reshape(X_train.shape[0],28,28)

#import matplotlib.pyplot as plt
#train_ori = train.reshape(train.shape[0],28,28)
#plt.imshow(train_ori[11])
def normalize(data):
    mean = data.mean(axis = 0)
    data -= mean
    std = data.std(axis = 0)
    data /=std
    return data,mean,std
#train,mean,std = normalize(train)
#test -= mean
#test /= std
#mean = train.mean().astype('float32')
#std = train.std().astype('float32')
#train = (train-mean) / std
from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(16,activation = 'relu', input_shape = (784,)))
model.add(layers.Dense(16,activation = 'relu'))
model.add(layers.Dense(10,activation = 'softmax'))

model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])

x_val = train[:9000]
y_val = train_label[:9000]

train_partial = train[9000:]
label_partial = train_label[9000:]

history = model.fit(train_partial,label_partial,epochs = 50,batch_size = 512,validation_data =(x_val,y_val))

#check the history contents
history_dict = history.history
history_dict.keys()
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1,len(acc) + 1)

plt.plot(epochs,loss,'bo',label = 'Training loss')
plt.plot(epochs, val_loss, 'b',label = 'Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.plot(epochs,acc,'bo',label = 'Training accuracy')
plt.plot(epochs, val_acc, 'b',label = 'Validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
predictions = model.predict_classes(test,verbose = 0)
submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                         "Label": predictions})
submissions.to_csv("DR.csv", index=False, header=True)