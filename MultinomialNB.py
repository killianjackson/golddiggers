import pickle

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

if __name__ == '__main__':
	main() 