import numpy as np
import tensorflow as tf
import tarfile
import os
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

gnb = GaussianNB();
def csv_to_numpy_array(filePath, delimiter):
    return np.genfromtxt(filePath, delimiter=delimiter, dtype=None)

def import_data():
    if "data" not in os.listdir(os.getcwd()):
        # Untar directory of data if we haven't already
        tarObject = tarfile.open("data.tar.gz")
        tarObject.extractall()
        tarObject.close()
        print("Extracted tar to current directory")
    else:
        # we've already extracted the files
        pass

    print("loading training data")
    trainX = csv_to_numpy_array("data/trainX.csv", delimiter="\t")
    trainY = csv_to_numpy_array("data/trainY.csv", delimiter="\t")
    trainY = trainY[:,0]
    print("loading test data")
    testX = csv_to_numpy_array("data/testX.csv", delimiter="\t")
    testY = csv_to_numpy_array("data/testY.csv", delimiter="\t")
    testY = testY[:,0]
    return trainX,trainY,testX,testY


###################
### IMPORT DATA ###
###################

trainX,trainY,testX,testY = import_data()

#clf = svm.SVC(C=0.1,kernel='linear')
svr = gnb.fit(trainX,trainY)

p = svr.predict(testX)
accuracy = accuracy_score(p,testY) * 100
print accuracy
