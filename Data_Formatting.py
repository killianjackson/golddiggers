TRAIN_FILE = "resources/train.json"
TEST_FILE = "resources/test.json"

from CuisineDB import *
from Matrix import *
import pickle

def main():
    db = CuisineDatabase(TRAIN_FILE, TEST_FILE)
    matrix_db = Matrix(db)
    (X, y) = matrix_db.trainingMatrix()
    pickle.dump(X, open("Matrix/X.p", "wb"))
    pickle.dump(y, open("Matrix/y.p", "wb"))

    (X_test) = matrix_db.testingMatrix()
    pickle.dump(X_test, open("Matrix/X_test.p", "wb"))
    
if __name__ == '__main__':
    main()
