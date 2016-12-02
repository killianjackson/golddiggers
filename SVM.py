import pickle
import json
import csv

from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

def main():
	X = pickle.load(open("Matrix/X.p", "rb"))
	y = pickle.load(open("Matrix/y.p", "rb"))

	print X
	print len(y)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

	# logreg = LogisticRegression(C=1e5, multi_class='multinomial', solver = "lbfgs")
	# logreg.fit(X_train, y_train)

	# clf = svm.SVC()
	clf = LinearSVC(verbose = 0, C = 0.13, intercept_scaling = 0.14)  # 0.1 for CountVectorizer
	clf.fit(X_train, y_train) 
	print "accuracy:"
	print clf.score(X_test, y_test)

	X_test_real = pickle.load(open("Matrix/X_test.p", "rb"))
	class_labels = clf.predict(X_test_real)
	testing_json = json.load(open("Resources/test.json"))

	with open('output_SVM.csv', 'w') as csvfile:
		fieldnames = ['id', 'cuisine']
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writeheader()
		for i in range(len(class_labels)):
			writer.writerow({'id': str(testing_json[i]["id"]), 'cuisine': class_labels[i]})

if __name__ == '__main__':
	main()    