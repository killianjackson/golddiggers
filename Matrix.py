import numpy as np
from scipy.sparse import coo_matrix

class Matrix(object):
    def __init__(self, cuisine_database):
        self._cdb = cuisine_database    #cuisine database
        self._num_ingr = 0                #ingredients count
        self._num_tr_rec = 0
        self._num_te_rec = 0
        self._ingr = {}                 #ingredients mapped to number

        # map ingredients into matrix
        for r in self._cdb.trainingSet():             #grab all sets of training recipes
            self._num_tr_rec += 1                     #increment training recipe count
            for i in r["ingredients"]:                #grab ingredients of training recipe
                if i not in self._ingr:               #if ingredient already exists, do nothing
                    self._ingr[i] = self._num_ingr    #else add ingredient and incremement number of ingredients
                    self._num_ingr += 1

        for r in self._cdb.testingSet():              #Do same for testing data
            self._num_te_rec += 1
            for i in r["ingredients"]:
                if i not in self._ingr:
                    self._ingr[i] = self._num_ingr
                    self._num_ingr += 1

    #https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.sparse.coo_matrix.html
    #Reference scipy.sparse.coo_matrix instantiation methods
    #https://docs.scipy.org/doc/numpy/reference/generated/numpy.ones.html
    #Reference numpy.ones for instantiated a multi-dimensional array of ones of float64 type

    def trainingMatrix(self):                         #n_tr_rec by n_ingr COOrdinate matrix
        classifications = []                          #classification vector
        rows = []                                     #COOrdinate matrix rows and columns vectors
        columns = []
        rec_num = 0
        for r in self._cdb.trainingSet():             #for each recipe in training data, and each ingredient
            for i in r["ingredients"]:
                rows.append(rec_num)                        #build rows and columns of coo sparse matrix with ingredients
                columns.append(self._ingr[i])
            classifications.append(r["cuisine"])      #build classification vector for each recipe
            rec_num += 1
        ndarray = np.ones((len(rows),), dtype=np.float64)
        matrix = coo_matrix((ndarray, (rows, columns)), shape=(self._num_tr_rec, self._num_ingr))
        # print "HERE"
        # print matrix, len(classifications)
        return (matrix, classifications)

    def testingMatrix(self):                          #n_te_rec by n_ingr COOrdinate matrix
        rows = []                                     #COOrdinate matrix rows and columns vectors
        columns = []
        rec_num = 0
        for r in self._cdb.testingSet():             #for each recipe in testing data, and each ingredient
            for i in r["ingredients"]:
                rows.append(rec_num)                        #build rows and columns of coo sparse matrix with ingredients
                columns.append(self._ingr[i])
            rec_num += 1
        ndarray = np.ones((len(rows),), dtype=np.float64)
        matrix = coo_matrix((ndarray, (rows, columns)), shape=(self._num_te_rec, self._num_ingr))
        return (matrix)