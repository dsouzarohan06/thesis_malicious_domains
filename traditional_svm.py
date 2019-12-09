#!/usr/bin/python

import numpy as np

file=open("/root/finaloutputs/1000_rows.csv","r")
X= []
label=[]
for line in file:
    Features=[]
    #print line #.split("\n")
    number_strings = line.split(',') # Split the line on runs of whitespace
    #print number_strings
    Features = number_strings[1:]   # Everything except for 1st element
    #print Features
    label.append(float(number_strings[0].strip()))
    numbers = [float(n) for n in Features] # Convert to float to get rid of newline chars
    #print numbers
    X.append(numbers) # Add the "row" to your list. X is features, Y is label

Y=label
#STEP 2:Partition The DataSet into Train and Test
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.4)

import time
start_time = time.time()
#from sklearn import linear_model
from sklearn import svm
#my_classifier2=linear_model.LogisticRegression(solver='lbfgs',max_iter=1000)
my_classifier2=svm.SVC(gamma='scale')
my_classifier2.fit(X_train,Y_train)

#STEP 5:Using Predict Method classify the Testing DataSet
predictions2=my_classifier2.predict(X_test)
#print("--- %s seconds ---" % (time.time() - start_time))
from sklearn.metrics import accuracy_score
print ("Accuracy is %s" % accuracy_score(Y_test,predictions2))

from sklearn.metrics import precision_score
prscore = precision_score(Y_test,predictions2, average='weighted')
print("Precision Score is %s" % prscore)

from sklearn.metrics import recall_score
rescore = recall_score(Y_test,predictions2, average='weighted')
print("Recall Score is %s" % rescore)

from sklearn.metrics import f1_score
f1score = f1_score(Y_test,predictions2, average='weighted')
print("F-1 Score is %s" % f1score)
print("--- %s seconds ---" % (time.time() - start_time))
