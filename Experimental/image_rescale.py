'''

https://hackernoon.com/visualizing-parts-of-convolutional-neural-networks-using-keras-and-cats-5cc01b214e59
https://stackoverflow.com/questions/43895750/keras-input-shape-for-conv2d-and-manually-loaded-images
'''

import matplotlib.pylab as plt
import matplotlib.image as mpimg
import numpy as np
import scipy
import keras

from scipy import misc
from keras.models import Sequential
from keras.layers import Conv2D



def show_cat(cat_batch):
    print("cat shape before transfo",cat_batch.shape)
    cat = np.squeeze(cat_batch,axis=0)
    print( "cat.shape", cat.shape)
    plt.imshow(cat)
    plt.show()

def resize_cat(cat):
    cat = scipy.misc.imresize(cat,size=(cat.shape[0]/2,cat.shape[1]/2))
    plt.imshow(cat)
    plt.show()

cat = mpimg.imread('cat.png')
print("Shape", cat.shape)
plt.imshow(cat)
plt.show()
resize_cat(cat)

cat_batch = cat.reshape(1,cat.shape[0],cat.shape[1],4)

input_shape = ( cat.shape[0], cat.shape[1], 4 )

model = Sequential()
model.add(Conv2D(4, kernel_size=(3, 3), activation='relu', input_shape=input_shape))

print("predicting ... ")
conv_cat = model.predict(cat_batch)
show_cat(conv_cat)
