import numpy as np
import CharProcessing
import random
import cPickle
import textparse
import theano

#This is number of characters which will be processed at once (in our project we used 15, 25 and 30)
SEQ_LENGTH = 15
N_BATCH = 100

class DataSet(object):

    X = None
    y = None
    mask = None
    charP = CharProcessing.CharProcessing()
    parser = textparse.TextParse()

    def getStuff(self):
        return (self.X.astype(theano.config.floatX), self.y.astype(theano.config.floatX),
            self.mask.astype(theano.config.floatX))

    def add(self, text, language):
        (newX, newY, newM) = self.parser.getInputMatrixFromText(text, SEQ_LENGTH, language)
        if self.X is not None:
            self.X = np.concatenate((self.X, newX), axis = 0)
        else:
            self.X = newX
        if self.y is not None:
            self.y = np.concatenate((self.y, newY), axis = 0)
        else:
            self.y = newY
        if self.mask is not None:
            self.mask = np.concatenate((self.mask, newM), axis = 0)
        else:
            self.mask = newM

    def batchCount (self):
        return len(self.X)/N_BATCH

    def getBatch(self, index):
        X1 = self.X[index*N_BATCH:(index + 1) * N_BATCH, :, :]
        y1 = self.y[index*N_BATCH:(index + 1) * N_BATCH, :]
        m1 = self.mask[index*N_BATCH:(index + 1) * N_BATCH, :]
        return (X1.astype(theano.config.floatX), y1.astype(theano.config.floatX),
            m1.astype(theano.config.floatX))

    def shuffle(self):
        tempX = None
        tempY = None
        tempM = None
        index_shuf = range(len(self.X))
        random.shuffle(index_shuf)
        for i in index_shuf:
            if tempX is None:
                tempX = np.zeros((1, self.X[i].shape[0], self.X[i].shape[1]))
                tempY = np.zeros((1, self.y[i].shape[0]))
                tempM = np.zeros((1, self.mask[i].shape[0]))
                tempX[0] = self.X[i]
                tempY[0] = self.y[i]
                tempM[0] = self.mask[i]
            else:
                tempX = np.concatenate((tempX, np.reshape(self.X[i], (1, self.X[i].shape[0], self.X[i].shape[1]))), axis = 0)
                tempY = np.concatenate((tempY, np.reshape(self.y[i], (1, self.y[i].shape[0]))), axis = 0)
                tempM = np.concatenate((tempM, np.reshape(self.mask[i], (1, self.mask[i].shape[0]))), axis = 0)
        self.X = tempX
        self.y = tempY
        self.mask = tempM


    def load (self, fileName):
        file = open(fileName, 'rb')
        self.X = cPickle.load(file)
        self.y = cPickle.load(file)
        self.mask = cPickle.load(file)
        #If your SEQ_LENGTH is 15, your mask should set mask[:, 15:30] to 0, if your SEQ_LENGTH is 25, you should set mask[:, 25:30] to 0
        #If your your SEQ_LENGTH is 30, you don't need mask
        self.mask[:, 15:30] = 0
        file.close()

    def store(self, fileName):
        file = open(fileName, 'wb')
        cPickle.dump(self.X, file)
        cPickle.dump(self.y, file)
        cPickle.dump(self.mask, file)
        file.close()