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
data = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train_data = (data.iloc[:,1:].values).astype('float32')
train_label = (data.iloc[:,0].values).astype('float32')
#train_label.shape
test_data = test.values.astype('float32')
from keras.utils import to_categorical
train_label = to_categorical(train_label)
#train_label.shape
train_data /= 255
test_data /= 255
from keras import layers
from keras import models
model = models.Sequential()
model.add(layers.Conv2D(32,(3,3), activation = 'relu',input_shape = (28,28,1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3), activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3), activation = 'relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.Dense(10,activation = 'softmax'))
train_data = train_data.reshape((train_data.shape[0],28,28,1))
test_data = test_data.reshape((test_data.shape[0],28,28,1))
model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.fit(train_data, train_label, epochs = 5,batch_size = 64)
predictions = model.predict_classes(test_data,verbose = 0)
submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                         "Label": predictions})
submissions.to_csv("DR.csv", index=False, header=True)