import random
import csv
from Matrix import *
from CuisineDB import *

#Results from submitting to Kaggle = 5.038%

class Classifier(object):
    def learn(self, ingredients, cuisine):
        raise NotImplementedError()
    def classify(self, ingredients):
        raise NotImplementedError()

class RandomClassifier(Classifier):

    def __init__(self):
        self._classes = set()

    def add_cuisine(self, cuisine):
        self._classes.add(cuisine)


def main():
    db = CuisineDatabase(TRAIN_FILE, TEST_FILE)
    matrix_db = Matrix(db)

    random_c = RandomClassifier()

    i = 0
    recipe = db.trainingDict(i)
    while (recipe != None):
        random_c.add_cuisine(recipe["cuisine"])
        i += 1
        recipe = db.trainingDict(i)

    testing_data = db.testingSet()

    with open('output_random.csv', 'w') as csvfile:
        fieldnames = ['id', 'cuisine']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(len(testing_data)):
            writer.writerow({'id': str(testing_data[i]["id"]), 'cuisine': random.sample(random_c._classes, 1)[0]})

if __name__ == '__main__':
    main()
