import pickle
import time

from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

def main():
	X = pickle.load(open("Matrix/X.p", "rb"))
	y = pickle.load(open("Matrix/y.p", "rb"))

	# print X
	# print len(y)
	for ratio in [0.1,0.3,0.5,0.7,0.9]:
		start_time = time.time()
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ratio, random_state=0)
		# logreg = LogisticRegression(C=1e5, multi_class='multinomial', solver = "lbfgs")
		# logreg.fit(X_train, y_train)	
		# clf = LinearSVC(verbose = 0)  # 0.1 for CountVectorizer
		# clf.fit(X_train, y_train) 
		# clf = RandomForestClassifier(n_estimators=20)
		# clf = clf.fit(X_train, y_train)
		nbc = MultinomialNB(alpha=0.19)
		nbc.fit(X_train, y_train)

		print (ratio, (time.time() - start_time))
	# logreg = LogisticRegression(C=1e5, multi_class='multinomial', solver = "lbfgs")
	# logreg.fit(X_train, y_train)

	# clf = svm.SVC()
	# clf = LinearSVC(verbose = 0)  # 0.1 for CountVectorizer
	# clf.fit(X_train, y_train) 
	# print "accuracy:"
	# print clf.score(X_test, y_test) 

if __name__ == '__main__':
	main()    