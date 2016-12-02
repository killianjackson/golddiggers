import json

class Db(object):
    def training_recipes(self, i):
        raise NotImplementedError()
    def testing_recipes(self, i):
        raise NotImplementedError()


class CuisineDatabase(Db):
    def __init__(self, training_filename, testing_filename):
        self._training_file = training_filename
        self._testing_file = testing_filename
        self._training_recipes = json.load(open(self._training_file))
        self._testing_recipes = json.load(open(self._testing_file))
        self.training_recipes_len = len(self._training_recipes)
        self.testing_recipes_len = len(self._testing_recipes)

    def trainingDict(self, i):
        if i < len(self._training_recipes):
            return self._training_recipes[i]
        else: return None

    def testingDict(self, i):
        if i < len(self._testing_recipes):
            return self._testing_recipes[i]
        else: return None

    def trainingSet(self):
        return self._training_recipes

    def testingSet(self):
        return self._testing_recipes
