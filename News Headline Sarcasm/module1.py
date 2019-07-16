import pandas as pd  
import numpy as np  
import pickle
from nltk.util import ngrams
from textblob import TextBlob
import re
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split 
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix  
from sklearn.metrics import accuracy_score

data = pd.read_csv("SVM_data.csv")
print(" Data Shape: ")
print(" -----------------------------------------------")
print(data.shape)
print()
print(" Data Header: ")
print(" -----------------------------------------------")
print(data.head())
print()

X = data.drop('Class', axis=1)  
y = data['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)

print(" Confusion Matrix: ")
print(" -----------------------------------------------")
print(confusion_matrix(y_test,y_pred))
print()
print(" Classification Report: ")
print(" -----------------------------------------------")
print(classification_report(y_test,y_pred))
print()
print(" Accuracy = ", end = "")
print (format(accuracy_score(y_test, y_pred), '.2f'))
print(" -----------------------------------------------")

filename = 'SVM_model.sav'
pickle.dump(svclassifier, open(filename, 'wb'))

#analyse = input("Enter your sarcasm: ")
#analyse = analyse.lower()
#analyse = re.sub(r'[^a-zA-Z0-9\s]', ' ', analyse)
#tokens = [token for token in analyse.split(" ") if token != ""]

#unigram_sum = 0
#bigram_sum = 0
#trigram_sum = 0
#total_sum = 0
#pos_high = 0
#pos_med = 0
#pos_low = 0
#neg_high = 0
#neg_med = 0
#neg_low = 0

#output = list(ngrams(tokens, 1))
#for z in range(len(output)):
#        #print(output[z])
#        blob = TextBlob(str(output[z]))
#        #print(blob.sentiment)
#        unigram_sum += blob.sentiment.polarity

#output = list(ngrams(tokens, 2))
#for z in range(len(output)):
#    #print(output[z])
#    blob = TextBlob(str(output[z]))
#    #print(blob.sentiment)
#    bigram_sum += blob.sentiment.polarity

#output = list(ngrams(tokens, 3))
#for z in range(len(output)):
#    #print(output[z])
#    blob = TextBlob(str(output[z]))
#    #print(blob.sentiment)
#    trigram_sum += blob.sentiment.polarity

#total_sum += unigram_sum + bigram_sum + trigram_sum
#if total_sum <= -1:
#    pos_low = 1
#    neg_high = 1
#elif total_sum >= 0 and total_sum <= 1:
#    pos_med = 1
#    neg_med = 1
#elif total_sum >= 2:            
#    pos_high = 1
#    neg_low = 1

#print(svclassifier.predict([[pos_high, pos_med, pos_low, neg_high, neg_med, neg_low]]))