from __future__ import print_function


import numpy as np
import theano
import theano.tensor as T
import lasagne
import DataSet
import CharProcessing
import random
import cPickle

# Number of units in the hidden (recurrent) layer
N_HIDDEN = 100
# Optimization learning rate
LEARNING_RATE = .001
# All gradients above this will be clipped
GRAD_CLIP = 100
# Number of epochs to train the net
NUM_EPOCHS = 250

def cost_function(predictedV, targetV):
    error = 0
    predictedMean = T.mean(predictedV, axis=1)
    targetMean = T.mean(targetV, axis=1)

    predictedIndex = T.argmax(predictedMean, axis=1)
    targetIndex = T.argmax(targetMean, axis=1)

    error = T.neq(predictedIndex, targetIndex).mean()
    return error

def confusionMatrix(predictedV, targetV):
    cv=np.zeros((CharProcessing.LANG_NO,CharProcessing.LANG_NO))
    predictedMean = np.mean(predictedV, axis = 1)
    targetMean = np.mean(targetV, axis = 1)
    predictedIndex = np.argmax(predictedMean, axis = 1)
    targetIndex = np.argmax(targetMean, axis = 1)
    for i in range(DataSet.N_BATCH):
        cv[predictedIndex[i], targetIndex[i]] += 1
    return cv

def recall(predictedV, targetV):
    error = 0
    predictedMean = T.mean(predictedV, axis=1)
    targetMean = T.mean(targetV, axis=1)

    predictedIndex = T.argmax(predictedMean, axis=1)
    targetIndex = T.argmax(targetMean, axis=1)

    error = T.neq(predictedIndex, targetIndex).mean()
    return error

def main(num_epochs=NUM_EPOCHS):
    #You should use next two lines if you want to work with validation set
    validationSet = DataSet.DataSet()
    validationSet.load("..\\Validation and training sets\\vSet_s_3_30.ds")

    #You should use next two lines if you want to work with training set
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

    print("l-forw shape: ")
    print(l_forward.output_shape)
    print("l-back shape: ")
    print(l_backward.output_shape)
    l_sum = lasagne.layers.ConcatLayer([l_forward, l_backward],axis=2)
    print("l-sum shape: ")
    print(l_sum.output_shape)
    l_reshape = lasagne.layers.ReshapeLayer(l_sum,(DataSet.N_BATCH*DataSet.SEQ_LENGTH, 2*N_HIDDEN))
    print("l-reshape shape: ")
    print(l_reshape.output_shape)
    l_out = lasagne.layers.DenseLayer(l_reshape, num_units=CharProcessing.LANG_NO,nonlinearity=lasagne.nonlinearities.softmax)
    print("l-out shape: ")
    print(l_out.output_shape)
    l_final = lasagne.layers.ReshapeLayer(l_out,(DataSet.N_BATCH,DataSet.SEQ_LENGTH,CharProcessing.LANG_NO))
    print("l-final shape: ")
    print(l_final.output_shape)

    target_values = T.tensor3('target_output')
    mask = T.matrix('mask')

    # lasagne.layers.get_output produces a variable for the output of the net
    predicted_values = lasagne.layers.get_output(l_final, l_in.input_var, mask=mask)
    predicted_numpy = theano.function([l_in.input_var, mask], predicted_values)
    # The value we care about is the final value produced for each sequence
    # Our cost will be mean-squared error
    cost = T.mean((predicted_values-target_values) ** 2)
    # Retrieve all parameters from the network
    all_params = lasagne.layers.get_all_params(l_final)
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
            #cM = np.zeros((CharProcessing.LANG_NO, CharProcessing.LANG_NO))
            #indexes = range(trainingSet.batchCount())
            #random.shuffle(indexes)
            #for bachNum in indexes:
            #    X, y, m = trainingSet.getBatch(bachNum)
            #    tempY=y.reshape((DataSet.N_BATCH,1,CharProcessing.LANG_NO))
            #    temp2Y=tempY
            #    for _ in range(DataSet.SEQ_LENGTH-1):
            #        tempY=np.concatenate((tempY,temp2Y),axis=1)
            #    y=tempY
            #    if X.shape[0] != DataSet.N_BATCH:
            #        continue
            #    train(X, y, m)
            #all_params_values = lasagne.layers.get_all_param_values(l_out)
            #f = open("trained-{}.nn".format(epoch), 'wb')
            #cPickle.dump(all_params_values, f, protocol=cPickle.HIGHEST_PROTOCOL)
            #f.close()

            #Code below is used to evaluate your model using validation set
            batchIndex = random.randint(0, validationSet.batchCount())
            X_val, y_val, mask_val = validationSet.getBatch(batchIndex)
            tmpY=y_val.reshape((DataSet.N_BATCH,1,CharProcessing.LANG_NO))
            tmp2Y=tmpY
            for _ in range(DataSet.SEQ_LENGTH-1):
                tmpY=np.concatenate((tmpY,tmp2Y),axis=1)
            y_val=tmpY
            cost_val = compute_cost(X_val, y_val, mask_val)

            print("Epoch {} validation cost = {}".format(epoch, cost_val))

            #Next lines make confusion matrix
            predicted = predicted_numpy(X_val, mask_val)
            updateCM = confusionMatrix(predicted, y_val)
            cM=np.add(cM,updateCM);
            f = open ("confusion-{}.txt".format(epoch), 'w')
            print("confusion:")
            for k in range(CharProcessing.LANG_NO):
                for l in range(CharProcessing.LANG_NO):
                    f.write(str(cM[k,l]))
                    f.write(' ')
                    print(str(cM[k,l]))
                print(' ')
                f.write('\n')
            f.close()
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    main()