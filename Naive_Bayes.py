import pickle
import json
import csv

from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB

def main():
	X = pickle.load(open("Matrix/X.p", "rb"))
	y = pickle.load(open("Matrix/y.p", "rb"))

#	print X
#	print len(y)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

	nbc = MultinomialNB(alpha=0.19)
#	nbc = BernoulliNB(alpha=0.09)

	nbc.fit(X_train, y_train)
	print "accuracy:"
	print nbc.score(X_test, y_test)

	X_test_real = pickle.load(open("Matrix/X_test.p", "rb"))
	class_labels = nbc.predict(X_test_real)
	testing_json = json.load(open("Resources/test.json"))

	with open('output_NB.csv', 'w') as csvfile:
		fieldnames = ['id', 'cuisine']
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writeheader()
		for i in range(len(class_labels)):
			writer.writerow({'id': str(testing_json[i]["id"]), 'cuisine': class_labels[i]})

if __name__ == '__main__':
	main()
