from __future__ import print_function

import numpy as np
import theano
import theano.tensor as T
import lasagne
import DataSet
import CharProcessing
import random
import cPickle

#main class

# Number of units in the hidden (recurrent) layer
N_HIDDEN = 100
# Optimization learning rate
LEARNING_RATE = .001
# All gradients above this will be clipped
GRAD_CLIP = 100
# Number of epochs to train the net
NUM_EPOCHS = 1000

def cost_function(predictedV, targetV):
    error = 0
    predictedIndex = T.argmax(predictedV, axis=1)
    targetIndex = T.argmax(targetV, axis=1)
    error = T.neq(predictedIndex, targetIndex).mean()
    return error

def confusionMatrix(predictedV, targetV):
    cv=np.zeros((CharProcessing.LANG_NO,CharProcessing.LANG_NO))
    predictedIndex = np.argmax(predictedV, axis = 1)
    targetIndex = np.argmax(targetV, axis = 1)
    for i in range(DataSet.N_BATCH):
        cv[predictedIndex[i], targetIndex[i]] += 1
    return cv

def main(num_epochs=NUM_EPOCHS):

    cM = np.zeros((CharProcessing.LANG_NO, CharProcessing.LANG_NO))

    #you can load validation or training set depends on what you want to achieve with the code
    validationSet = DataSet.DataSet()
    validationSet.load("..\\Validation and training sets\\vSet_s_3_30.ds")

    #trainingSet = DataSet.DataSet()
    #trainingSet.load("..\\Validation and training sets\\tSet_s_3_30.ds")

    print("Building network ")
    
    charP = CharProcessing.CharProcessing()
    # First, we build the network, starting with an input layer
    # Recurrent layers expect input of shape
    # (batch size, max sequence length, number of features)
    l_in = lasagne.layers.InputLayer(shape=(DataSet.N_BATCH, DataSet.SEQ_LENGTH, charP.getSupportedNoOfLetters()))
    # We're using a bidirectional network, which means we will combine two
    # RecurrentLayers, one with the backwards=True keyword argument.
    # Setting a value for grad_clipping will clip the gradients in the layer

    #LSTM architecture
    l_forward = lasagne.layers.LSTMLayer(l_in, N_HIDDEN, backwards=False,
                                         learn_init=True, peepholes=True)

    l_backward = lasagne.layers.LSTMLayer(l_in, N_HIDDEN, backwards=True,
                                          learn_init=True, peepholes=True)


    #RNN architecture
    #l_forward = lasagne.layers.RecurrentLayer(
    #    l_in, N_HIDDEN, grad_clipping=GRAD_CLIP,
    #    W_in_to_hid=lasagne.init.HeUniform(),
    #    W_hid_to_hid=lasagne.init.HeUniform(),
    #    nonlinearity=lasagne.nonlinearities.tanh)
    #l_backward = lasagne.layers.RecurrentLayer(
    #    l_in, N_HIDDEN, grad_clipping=GRAD_CLIP,
    #    W_in_to_hid=lasagne.init.HeUniform(),
    #    W_hid_to_hid=lasagne.init.HeUniform(),
    #    nonlinearity=lasagne.nonlinearities.tanh, backwards=True)
    
    # The objective of this task depends only on the final value produced by
    # the network.  So, we'll use SliceLayers to extract the LSTM layer's
    # output after processing the entire input sequence.  For the forward
    # layer, this corresponds to the last value of the second (sequence length)
    # dimension.
    l_forward_slice = lasagne.layers.SliceLayer(l_forward, -1, 1)
    # For the backwards layer, the first index actually corresponds to the
    # final output of the network, as it processes the sequence backwards.
    l_backward_slice = lasagne.layers.SliceLayer(l_backward, 0, 1)
    # Now, we'll concatenate the outputs to combine them.
    l_sum = lasagne.layers.ConcatLayer([l_forward_slice, l_backward_slice])
    # Our output layer is a simple dense connection, with 1 output unit
    l_out = lasagne.layers.DenseLayer(l_sum, num_units=CharProcessing.LANG_NO,
        nonlinearity=lasagne.nonlinearities.softmax)

    target_values = T.matrix('target_output')
    mask = T.matrix('mask')

    # lasagne.layers.get_output produces a variable for the output of the net
    predicted_values = lasagne.layers.get_output(l_out, l_in.input_var, mask=mask)
    predicted_numpy = theano.function([l_in.input_var, mask], predicted_values)
    # The value we care about is the final value produced for each sequence
    # Our cost will be mean-squared error
    cost = T.mean((predicted_values-target_values) ** 2)
    # Retrieve all parameters from the network
    all_params = lasagne.layers.get_all_params(l_out)
    # Compute SGD updates for training
    print("Computing updates ...")
    updates = lasagne.updates.adagrad(cost, all_params, LEARNING_RATE)
    # Theano functions for training and computing cost
    print("Compiling functions ...")
    train = theano.function([l_in.input_var, target_values, mask], cost, updates=updates)
    error_rate = cost_function(predicted_values, target_values)
    compute_cost = theano.function([l_in.input_var, target_values, mask], error_rate)

    #If you are working with training set these three lines should be commented
    f = open("trained.nn", 'rb')
    lasagne.layers.set_all_param_values(l_out, cPickle.load(f))
    f.close()

    print("Training ...")
    try:
        for epoch in range(num_epochs):

            #Commented code below should be used if you want to train data
            #indexes = range(trainingSet.batchCount())
            #random.shuffle(indexes)
            #for bachNum in indexes:
            #    X, y, m = trainingSet.getBatch(bachNum)
            #    if X.shape[0] != DataSet.N_BATCH:
            #        continue
            #    train(X, y, m)
            #all_params_values = lasagne.layers.get_all_param_values(l_out)
            #f = open("trained.nn", 'wb')
            #cPickle.dump(all_params_values, f, protocol=cPickle.HIGHEST_PROTOCOL)
            #f.close()

            #Code below is used to evaluate your model using validation set
            batchIndex = random.randint(0, validationSet.batchCount())
            X_val, y_val, mask_val = validationSet.getBatch(batchIndex)
            cost_val = compute_cost(X_val, y_val, mask_val)
            print("Epoch {} validation cost = {}".format(epoch, cost_val))
            predicted = predicted_numpy(X_val, mask_val)
            updateCM = confusionMatrix(predicted, y_val)
            cM=np.add(cM,updateCM)
        f = open ("confusion.txt", 'w')
        for k in range(CharProcessing.LANG_NO):
            for l in range(CharProcessing.LANG_NO):
                f.write(str(cM[k,l]))
                f.write(' ')
            f.write('\n')
        f.close()
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()
