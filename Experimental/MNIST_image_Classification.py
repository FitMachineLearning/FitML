'''

MNIST dataset classification with Keras.
Credits to fchollet. Find him on github.
'''

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

batch_size = 128
num_class = 10
epochs = 3

#input image dimensions
img_rows, img_cols = 28,28

#seperate train and test dataset
(x_train,y_train), (x_test,y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0],1,img_rows,img_cols)
    x_test = x_test.reshape(x_test.shape[0],1,img_rows,img_cols)
    input_shape = (1,img_rows,img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0],img_rows,img_cols,1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows,img_cols,1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /=255
x_test /=255
print('x_train shape', x_train.shape)
print('x_test shape', x_test.shape)
print('y_train', y_train)


# convert class to binanry class
y_train = keras.utils.to_categorical(y_train,num_class)
y_test = keras.utils.to_categorical(y_test,num_class)

#print('y_train', y_train)

#Declare the model
model = Sequential()
model.add(Conv2D(32,kernel_size=(3,3), activation='relu',input_shape=input_shape))
model.add(Conv2D(64,(3,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add((Dropout(0.5)))
model.add(Dense(num_class,activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train,y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(x_test,y_test))

score = model.evaluate(x_test[1000:],y_test[1000:],verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
