TRAIN_FILE = "resources/train.json"
TEST_FILE = "resources/test.json"

from CuisineDB import *
from Matrix import *

def main():
    db = CuisineDatabase(TRAIN_FILE, TEST_FILE)
    matrix_db = Matrix(db)

if __name__ == '__main__':
    main()