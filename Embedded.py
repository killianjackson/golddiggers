import pickle
import json
import csv

from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def main():
	X = pickle.load(open("Matrix/X.p", "rb"))
	y = pickle.load(open("Matrix/y.p", "rb"))
	X_test_real = pickle.load(open("Matrix/X_test.p", "rb"))
	testing_json = json.load(open("Resources/test.json"))

	#print X
	#print len(y)
	#print y

	#split the data into single method training set, single method testing set(=embedded method training set), and embedded method testing set
	X_embedded_temp,X_embedded_test,y_embedded_temp,y_embedded_test = train_test_split(X, y, test_size = 0.05, random_state = 0)
	X_single_train,X_single_test,y_single_train,y_single_test = train_test_split(X_embedded_temp,y_embedded_temp,test_size = 0.1,random_state = 0)


	#SVM method No1
	clf = LinearSVC(verbose = 0, C = 0.13, intercept_scaling = 0.14)
	clf.fit(X_single_train, y_single_train)

	labels_on_single_test_svm = clf.predict(X_single_test)
	labels_on_embedded_test_svm =  clf.predict(X_embedded_test)
	labels_on_real_test_svm = clf.predict(X_test_real)
	print "SVM finished."



    #Naivebayes No2
	nbc = MultinomialNB(alpha=0.19)
	nbc.fit(X_single_train, y_single_train)

	labels_on_single_test_nb = clf.predict(X_single_test)
	labels_on_embedded_test_nb = clf.predict(X_embedded_test)
	labels_on_real_test_nb = clf.predict(X_test_real)
	print "Naivebayes finished."


    #MultinomialNB No3
	mnb = MultinomialNB(alpha=1.15, fit_prior=False, class_prior=None)
	mnb.fit(X_single_train, y_single_train)
	labels_on_single_test_mb = clf.predict(X_single_test)
	labels_on_embedded_test_mb = clf.predict(X_embedded_test)
	labels_on_real_test_mb = clf.predict(X_test_real)
	print "MultinomialNB finished."



	#logreg method No4
	logreg = LogisticRegression(C=1e5, multi_class='multinomial', solver = "lbfgs")
	logreg.fit(X_single_train, y_single_train)

	labels_on_single_test_logreg = logreg.predict(X_single_test)
	labels_on_embedded_test_logreg = logreg.predict(X_embedded_test)
	labels_on_real_test_logreg = logreg.predict(X_test_real)
	print "LogisticRegression finished."



	#RandomForest No5
	clf = RandomForestClassifier(n_estimators=10)
	clf = clf.fit(X_single_train, y_single_train)

	labels_on_single_test_rf = logreg.predict(X_single_test)
	labels_on_embedded_test_rf = logreg.predict(X_embedded_test)
	labels_on_real_test_rf = logreg.predict(X_test_real)
	print "RandomForest finished."



	classifierLabel_single_test = []
	classifierLabel_embedded_test = []
	classifierLabel_real_test = []


	#produce the embedded classification label for every case in single method testing set
	#single method testing set is used as embedded method training set here
	for i in range(0,len(y_single_test)):
		if labels_on_single_test_svm[i]==y_single_test[i]:
			classifierLabel_single_test.append(1)
			continue
		if labels_on_single_test_nb[i]==y_single_test[i]:
			classifierLabel_single_test.append(2)
			continue
		if labels_on_single_test_mb[i] == y_single_test[i]:
			classifierLabel_single_test.append(3)
			continue
		if labels_on_single_test_logreg[i] == y_single_test[i]:
			classifierLabel_single_test.append(4)
			continue
		if labels_on_single_test_rf[i] == y_single_test[i]:
			classifierLabel_single_test.append(5)
			continue
		#if none of the classifier predicts right we arbritary choose one( use the best one here)
		classifierLabel_single_test.append(1)



	#produce the embedded classification label for every case in embedded method testing set
	for i in range(0,len(y_embedded_test)):
		if labels_on_embedded_test_svm[i]==y_embedded_test[i]:
			classifierLabel_embedded_test.append(1)
			continue
		if labels_on_embedded_test_nb[i]==y_embedded_test[i]:
			classifierLabel_embedded_test.append(2)
			continue
		if labels_on_embedded_test_mb[i] == y_embedded_test[i]:
			classifierLabel_embedded_test.append(3)
			continue
		if labels_on_embedded_test_logreg[i] == y_embedded_test[i]:
			classifierLabel_embedded_test.append(4)
			continue
		if labels_on_embedded_test_rf[i] == y_embedded_test[i]:
			classifierLabel_embedded_test.append(5)
			continue
		#if none of the classifier predicts right we arbritary choose one( use the best one here)
		classifierLabel_embedded_test.append(1)


	#print "Single---------------------------"
	#for item in classifierLabel_single_test:
	#	print item

	#print "Embedded-------------------------"

	#for item in classifierLabel_embedded_test:
	#	print item




	#build the final embedded method classifier
	clf_embedded = LinearSVC(verbose = 0, C = 0.4, intercept_scaling = 0.14)
	#the training stage of the embedded classifier
	clf_embedded.fit(X_single_test,classifierLabel_single_test)

	Predicty_on_embedded_test= clf_embedded.predict(X_embedded_test)


	Prediction_on_embedded_test = []
	for i in range(0,len(Predicty_on_embedded_test)):
		if Predicty_on_embedded_test[i]==1:
			Prediction_on_embedded_test.append(labels_on_embedded_test_svm[i])
			continue
		if Predicty_on_embedded_test[i]==2:
			Prediction_on_embedded_test.append(labels_on_embedded_test_nb[i])
			continue
		if Predicty_on_embedded_test[i]==3:
			Prediction_on_embedded_test.append(labels_on_embedded_test_mb[i])
			continue
		if Predicty_on_embedded_test[i]==4:
			Prediction_on_embedded_test.append(labels_on_embedded_test_logreg[i])
			continue
		Prediction_on_embedded_test.append(labels_on_embedded_test_rf[i])


	hit = 0
	for i in range(0,len(Prediction_on_embedded_test)):
		if Prediction_on_embedded_test[i] == y_embedded_test[i]:
			hit = hit + 1


	print hit/(len(y_embedded_test)+0.0)


	# predict the classifier used for every real testing set
	Predicty_on_real_test = clf_embedded.predict(X_test_real)

	Prediction_on_real_test =[]
	for i in range(0,len(Predicty_on_real_test)):
		if Predicty_on_real_test[i]==1:
			Prediction_on_real_test.append(labels_on_real_test_svm[i])
			continue
		if Predicty_on_real_test[i]==2:
			Prediction_on_real_test.append(labels_on_real_test_nb[i])
			continue
		if Predicty_on_real_test[i]==3:
			Prediction_on_real_test.append(labels_on_real_test_mb[i])
			continue
		if Predicty_on_real_test[i]==4:
			Prediction_on_real_test.append(labels_on_real_test_logreg[i])
			continue
		Prediction_on_real_test.append(labels_on_real_test_rf[i])


	with open('output_embedded.csv', 'w') as csvfile:
		fieldnames = ['id', 'cuisine']
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writeheader()
		for i in range(len(Prediction_on_real_test)):
			writer.writerow({'id': str(testing_json[i]["id"]), 'cuisine': Prediction_on_real_test[i]})



		





if __name__ == '__main__':
	main()  


