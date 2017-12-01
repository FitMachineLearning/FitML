# coding: utf-8

# # Annotations for the Sirajology Python NN Example
#
# This code comes from a demo NN program from the YouTube video https://youtu.be/h3l4qz76JhQ. The program creates an neural network that simulates the exclusive OR function with two inputs and one output.
#
#

# In[23]:

import numpy as np  # Note: there is a typo on this line in the video
import pandas as pd
import os


# The following is a function definition of the sigmoid function, which is the type of non-linearity chosen for this neural net. It is not the only type of non-linearity that can be chosen, but is has nice analytical features and is easy to teach with. In practice, large-scale deep learning systems use piecewise-linear functions because they are much less expensive to evaluate.
#
# The implementation of this function does double duty. If the deriv=True flag is passed in, the function instead calculates the derivative of the function, which is used in the error backpropogation step.

# In[24]:


def nonlin(x, deriv=False):  # Note: there is a typo on this line in the video
    if(deriv==True):
        return (x*(1-x))

    return 1/(1+np.exp(-x))  # Note: there is a typo on this line in the video


# The following code creates the input matrix. Although not mentioned in the video, the third column is for accommodating the bias term and is not part of the input.

# In[25]:

def read_inputs_and_outputs():
    fullDataSet = pd.read_csv(dir_path + "/sonar.csv")
    #load input data
    X = fullDataSet[fullDataSet.columns[0:60]].values
    Y_ = fullDataSet[fullDataSet.columns[60]]
    Y = np.zeros((X.shape[0],1))

    print("X.shape", X.shape)
    print("Y.shape", Y.shape)
    print(Y[0:2])

    for i in range(X.shape[0]):
        if Y_[i] == 'R':
            Y[i,0] = 1
        else:
            Y[i,0] = 0

    return (X, Y)


dir_path = os.path.dirname(os.path.realpath(__file__))

X, Y = read_inputs_and_outputs()
#input data
#X = np.array([[0,0,1],  # Note: there is a typo on this line in the video
#            [0,1,1],
#            [1,0,1],
#            [1,1,1]])


# The output of the exclusive OR function follows.

# In[26]:

#output data
#y = np.array([[0],
#             [1],
#             [1],
#             [0]])


# The seed for the random generator is set so that it will return the same random numbers each time, which is sometimes useful for debugging.

# In[27]:

np.random.seed(6)


# Now we intialize the weights to random values. syn0 are the weights between the input layer and the hidden layer.  It is a 3x4 matrix because there are two input weights plus a bias term (=3) and four nodes in the hidden layer (=4). syn1 are the weights between the hidden layer and the output layer. It is a 4x1 matrix because there are 4 nodes in the hidden layer and one output. Note that there is no bias term feeding the output layer in this example. The weights are initially generated randomly because optimization tends not to work well when all the weights start at the same value. Note that neither of the neural networks shown in the video describe the example.

#Declare constants

num_nodes_input_layer = 60
num_nodes_hl1 = 60
num_nodes_hl2 = 60
num_nodes_output_layer = 1
learning_rate = 0.005


# In[28]:

#synapses
syn0 = 2*np.random.random((num_nodes_input_layer,num_nodes_hl1)) - 1  # 3x4 matrix of weights ((2 inputs + 1 bias) x 4 nodes in the hidden layer)
syn1 = 2*np.random.random((num_nodes_hl1, num_nodes_hl2)) - 1  # 4x1 matrix of weights. (4 nodes x 1 output) - no bias term in the hidden layer.
syn2 = 2*np.random.random((num_nodes_hl2,num_nodes_output_layer)) -1

print("syn2 = ", syn2.shape)
print(syn2[0:10])


# This is the main training loop. The output shows the evolution of the error between the model and desired. The error steadily decreases.

# In[29]:

#training step
# Python2 Note: In the follow command, you may improve
#   performance by replacing 'range' with 'xrange'.
for j in range(3000):

    # Calculate forward through the network.
    l0 = X
    l1 = nonlin(np.dot(l0, syn0))
    l2 = nonlin(np.dot(l1, syn1))
    l_out = nonlin(np.dot(l2, syn2))



    # Back propagation of errors using the chain rule.
    l_out_error = Y - l_out
    cost = np.sum(l_out_error**2)/2

    if j == 0 :
        print("layer out output", l_out.shape)
        print(l_out)

    print("ittaration ", j, " overall cost = ", cost)


    if j  == 0 or j==1:   # Only print the error every 10000 steps, to save time and limit the amount of output.
        print("layer error", l_out_error)
        print(l_out_error)



    if j  == 0 or j==1:   # Only print the error every 10000 steps, to save time and limit the amount of output.
        print("Error: " + str(np.mean(np.abs(l_out_error))))

    l_out_delta = l_out_error*nonlin(l_out, deriv=True)
    if j == 0 or j==1 :
        print("layer 1 output", l1.shape)
        #print(l1)


    l2_error = l_out_delta.dot(syn2.T)
    l2_delta = l2_error * nonlin(l2,deriv=True)
    if j == 0 or j==1 :
        print("l2 ")
        #print(l2)



    l1_error = l2_delta.dot(syn1.T)

    l1_delta = l1_error * nonlin(l1,deriv=True)

    #update weights (no learning rate term)
    syn2 += learning_rate * l2.T.dot(l_out_delta)
    syn1 += learning_rate * l1.T.dot(l2_delta)
    syn0 += learning_rate * l0.T.dot(l1_delta)

print("Output after training")
print(l_out[80:120])




# See how the final output closely approximates the true output [0, 1, 1, 0]. If you increase the number of interations in the training loop (currently 60000), the final output will be even closer.

# In[30]:

#get_ipython().run_cell_magic(u'HTML', u'', u'#The following line is for embedding the YouTube video \n#   in this Jupyter Notebook. You may remove it without peril. \n<iframe width="560" height="315" src="https://www.youtube.com/embed/h3l4qz76JhQ" frameborder="0" allowfullscreen></iframe>')


# In[ ]:
