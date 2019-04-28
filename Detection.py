from __future__ import division
import sys
import os
import time
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.cross_validation import KFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import accuracy_score
from sklearn.linear_model import SGDClassifier

def readFile(filename,label):
    #print filename
    reviewDict = {}
    reviewList = []
    reviewLabel = []
    if label == 1:
        value =1
    else:
        value =0 
    count = 0
    with open(filename) as f:
        for line in f:
            review = line.split()
            reviewID = review[0]
            #print reviewID
            reviewContent = " ".join(review[1:])
            reviewList.append(reviewContent)
            reviewLabel.append(value)
            reviewDict[reviewContent] = reviewID

            #print reviewID, reviewContent,"EOR"
           # print reviewContent

    #print reviewDict
    return (reviewDict,reviewList,reviewLabel)

def accuracy(array):
    count = 0
    correct = 0
    for element in array:
        count =count +1
        if(count<=10):
            if(element == 0):
                correct = correct + 1
        if(count >10):
            if(element == 1):
                correct = correct + 1
    print correct/20

def Testing(train_data, train_labels, test_data,testreviewDict):

    sentiment = ""
    fo = open("outputSVM.txt","a")
    size = len(test_data) - 1
    IDList = []
    train_data_np = np.asarray(train_data)
    train_labels_np = np.asarray(train_labels)
    test_data = np.asarray(test_data) 

    vectorizer = Pipeline([("Countvector",CountVectorizer(ngram_range=(1,1),stop_words='english')),("Chi Square",SelectKBest(chi2, k=15)),("svm",SVC(C=3.0))]); 
    vectorizer.fit(train_data_np,train_labels_np)
    test_vectors = vectorizer.predict(test_data)
    for element in test_data:
        IDList.append(testreviewDict[element])
    while(size>0):
        #print test_vectors[size]
        if(test_vectors[size] == 1):
            sentiment = "T"
        else:
            sentiment = "F"
        print IDList[size],sentiment
        fo.write(IDList[size]+"\t"+sentiment+"\n")

        size = size - 1

#Logistic Regression
def RegressionClassifier(training_data,training_list,testing_data):
    accuracy = []
    correctsum = 0
    kf = KFold(len(training_data), n_folds=300)
    for train_data,test_data in kf:
        xtrain = np.asarray(training_data)[train_data]
       # print "xtrain",xtrain
        xtest = np.asarray(training_data)[test_data]
        #print "xtest:",xtest
        ytrain = np.asarray(training_list)[train_data]
        #print ytrain
        ytest = np.asarray(training_list)[test_data]
       # print ytest
    #train_data_np = np.asarray(training_data)
    #train_labels_np = np.asarray(training_list)
    #test_data_np = np.asarray(testing_data)
        vectorizer = CountVectorizer(max_df=1.0, max_features=None, min_df=1, ngram_range=(1, 1), preprocessor=None, stop_words='english', strip_accents=None, tokenizer=None, vocabulary=None)
        train_vectors = vectorizer.fit_transform(xtrain)
        test_vectors = vectorizer.transform(xtest)


        logit = LogisticRegression(C=1.0).fit(train_vectors,ytrain)
        predict_vectors = logit.predict(test_vectors)
        accuracy.append(accuracy_score(ytest,predict_vectors))
    print accuracy
    for element in accuracy:
        correctsum = correctsum + element
    print correctsum/len(accuracy)
#using Leave-one-out validation technique:

def SVMClassifier2(training_data,training_list,testing_data):
    #print len(training_data)
    accuracy = []
    correct = 0
    count = 0
    leaveOneOut = cross_validation.LeaveOneOut(len(training_data)) 
    #print "LOO:",leaveOneOut
    #print "here"
    for train_data,test_data in leaveOneOut:
        xtrain = np.asarray(training_data)[train_data]
       # print "xtrain",xtrain
        xtest = np.asarray(training_data)[test_data]
        #print "xtest:",xtest
        ytrain = np.asarray(training_list)[train_data]
        ytest = np.asarray(training_list)[test_data]
        count = count + 1
        #train_labels=combinedlabels
       # print count
        vectorizer = Pipeline([("Countvector",CountVectorizer(ngram_range=(1,1),stop_words='english')),("Chi Square",SelectKBest(chi2, k=15)),("svm",SVC(C=3.0))]); 
        vectorizer.fit(xtrain,ytrain)
        test_vectors = vectorizer.predict(xtest)
        #print test_vectors
        accuracy.append(accuracy_score(ytest,test_vectors))

    #print "hereee"
    #print accuracy
    for score in accuracy:
        if score == 1:
            correct = correct + 1
    print "Accuracy - Leave one out cross validation:", (correct/len(training_data))

# 90-10 split
def SVMClassifier3(train_data, train_labels, test_data):

    accuracy = []
    correct = 0

    train_data_np = np.asarray(train_data)
    train_labels_np = np.asarray(train_labels)
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(train_data_np, train_labels_np , test_size=0.1, random_state=4)
    #print len(X_train),len(X_test),len(y_train),len(y_test)
    vectorizer = Pipeline([("Countvector",CountVectorizer(ngram_range=(1,1),stop_words='english')),("Chi Square",SelectKBest(chi2, k=15)),("svm",SVC(C=4.0))]); 
    vectorizer.fit(X_train,y_train)
    test_vectors = vectorizer.predict(X_test)
    print "Accuracy with 90-10 split:",accuracy_score(y_test,test_vectors)
    #print(classification_report(y_test, test_vectors))

#k-fold cross validation
def SVMClassifier4(training_data,training_list,testing_data):
    accuracy = []
    sumscore =0 
    kf = KFold(len(training_data), n_folds=300)
    for train_data,test_data in kf:
        xtrain = np.asarray(training_data)[train_data]
       # print "xtrain",xtrain
        xtest = np.asarray(training_data)[test_data]
        #print "xtest:",xtest
        ytrain = np.asarray(training_list)[train_data]
        ytest = np.asarray(training_list)[test_data]
     #   count = count + 1
        #train_labels=combinedlabels
       # print count
        vectorizer = Pipeline([("Countvector",CountVectorizer(ngram_range=(1,1),stop_words='english')),("Chi Square",SelectKBest(chi2, k=15)),("svm",SVC(C=3.0))]); 
        vectorizer.fit(xtrain,ytrain)
        test_vectors = vectorizer.predict(xtest)
        #print test_vectors
        accuracy.append(accuracy_score(ytest,test_vectors))
    for score in accuracy:
        sumscore = sumscore+ score
    print "Accuracy: KFold Cross validation",sumscore/len(accuracy)
    #print accuracy

# 3 different classifiers 
def SVMClassifier(train_data, train_labels, test_data):

    train_data_np = np.asarray(train_data)
    train_labels_np = np.asarray(train_labels)
 

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(train_data_np, train_labels_np , test_size=0.2, random_state=1)
    print len(X_train),len(X_test),len(y_train),len(y_test)

    vectorizer = CountVectorizer(max_df=1.0, max_features=None, min_df=1, ngram_range=(1, 1), preprocessor=None, stop_words='english', strip_accents=None, tokenizer=None, vocabulary=None)
    train_vectors = vectorizer.fit_transform(X_train)
    test_vectors = vectorizer.transform(X_test)


    #print train_vectors
   
 # Perform classification with SVM, kernel=rbf
    classifier_rbf = svm.SVC()
    t0 = time.time()
    classifier_rbf.fit(train_vectors, y_train)
    t1 = time.time()
    prediction_rbf = classifier_rbf.predict(test_vectors)
    print "RBF:",prediction_rbf
    #accuracy(prediction_rbf)
    t2 = time.time()
    time_rbf_train = t1-t0
    time_rbf_predict = t2-t1
    print("Results for SVC(kernel=rbf)")
    print("Training time: %fs; Prediction time: %fs" % (time_rbf_train, time_rbf_predict))
    print(classification_report(y_test, prediction_rbf))

    #Perform classification with SVM, kernel=linear
    classifier_linear = svm.SVC(kernel='linear')
    t0 = time.time()
    classifier_linear.fit(train_vectors, y_train)
    t1 = time.time()
    prediction_linear = classifier_linear.predict(test_vectors)
    print "Linear:",prediction_linear
   # accuracy(prediction_linear)
    t2 = time.time()
    time_linear_train = t1-t0
    time_linear_predict = t2-t1
    print("Results for SVC(kernel=linear)")
    print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
    print(classification_report(y_test, prediction_linear))

    # Perform classification with SVM, kernel=linear
    classifier_liblinear = svm.LinearSVC()
    t0 = time.time()
    classifier_liblinear.fit(train_vectors, y_train)
    t1 = time.time()
    prediction_liblinear = classifier_liblinear.predict(test_vectors)
    print "LibLibear:",prediction_liblinear
   # accuracy(prediction_liblinear)

    t2 = time.time()
    time_liblinear_train = t1-t0
    time_liblinear_predict = t2-t1
    print("Results for LinearSVC()")
    print("Training time: %fs; Prediction time: %fs" % (time_liblinear_train, time_liblinear_predict))
    print(classification_report(y_test, prediction_liblinear))

    #train_data_np = np.asarray(train_vectors)
    
    #test_data_np = np.asarray(test_vectors)
    #print test_data
    #clf = svm.SVC(kernel='linear', C=1).fit(train_vectors, y_train)
    
    #scores = cross_validation.cross_val_score(clf, test_vectors, y_test, cv=4)
    #print "Score:",clf.score(test_vectors, y_test) 
    #print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    # Print results in a nice table
    
    
   
def driver():
    count = 0
    reviewDict = {}
    reviewLabel = []
    reviewList = []
    truereviewDict,truereviewList,truereviewLabel = readFile('hotelT-train.txt',1)
    #print truereviewList
    falsereviewDict,falsereviewList,falsereviewLabel = readFile('hotelF-train.txt',0)
    testreviewDict,testreviewList,testreviewLabel = readFile('test.txt',0)

    reviewList = truereviewList + falsereviewList
    reviewLabel = truereviewLabel + falsereviewLabel
    
    #SVMClassifier(reviewList,reviewLabel,testreviewList)
    #SVMClassifier2(reviewList,reviewLabel,testreviewList)
    #SVMClassifier3(reviewList,reviewLabel,testreviewList)
    #SVMClassifier4(reviewList,reviewLabel,testreviewList)
    #RegressionClassifier(reviewList,reviewLabel,testreviewList)   
    Testing(reviewList,reviewLabel,testreviewList,testreviewDict)
driver()
