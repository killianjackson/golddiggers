import pickle
import json
import csv

from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

def main():
	X = pickle.load(open("Matrix/X.p", "rb"))
	y = pickle.load(open("Matrix/y.p", "rb"))

	print X
	print len(y)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

	mnb = MultinomialNB(alpha=1.15, fit_prior=False, class_prior=None)
	mnb.fit(X_train, y_train) 
	print "accuracy:"
	print mnb.score(X_test, y_test)

	X_test_real = pickle.load(open("Matrix/X_test.p", "rb"))
	class_labels = mnb.predict(X_test_real)
	testing_json = json.load(open("Resources/test.json"))

	with open('output_mnb.csv', 'w') as csvfile:
		fieldnames = ['id', 'cuisine']
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writeheader()
		for i in range(len(class_labels)):
			writer.writerow({'id': str(testing_json[i]["id"]), 'cuisine': class_labels[i]})

if __name__ == '__main__':
	main() 