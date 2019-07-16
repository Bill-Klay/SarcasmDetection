import pickle
import re
from nltk.util import ngrams
from textblob import TextBlob

filename = 'SVM_model.sav'
svclassifier = pickle.load(open(filename, 'rb'))

analyse = input("Enter your sarcasm: ")
while analyse != "exit":
    analyse = analyse.lower()
    analyse = re.sub(r'[^a-zA-Z0-9\s]', ' ', analyse)
    tokens = [token for token in analyse.split(" ") if token != ""]

    unigram_sum = bigram_sum = trigram_sum = total_sum = pos_high = pos_med = pos_low = neg_high = neg_med = neg_low = 0

    output = list(ngrams(tokens, 1))
    for z in range(len(output)):
        #print(output[z])
        blob = TextBlob(str(output[z]))
        #print(blob.sentiment)
        unigram_sum += blob.sentiment.polarity

    output = list(ngrams(tokens, 2))
    for z in range(len(output)):
        #print(output[z])
        blob = TextBlob(str(output[z]))
        #print(blob.sentiment)
        bigram_sum += blob.sentiment.polarity

    output = list(ngrams(tokens, 3))
    for z in range(len(output)):
        #print(output[z])
        blob = TextBlob(str(output[z]))
        #print(blob.sentiment)
        trigram_sum += blob.sentiment.polarity

    total_sum += unigram_sum + bigram_sum + trigram_sum
    if total_sum <= -1:
        pos_low = 1
        neg_high = 1
    elif total_sum >= 0 and total_sum <= 1:
        pos_med = 1
        neg_med = 1
    elif total_sum >= 2:           
        pos_high = 1
        neg_low = 1

    print(svclassifier.predict([[pos_high, pos_med, pos_low, neg_high, neg_med, neg_low]]))
    percent = (total_sum/3)
    if percent < 0:
        percent *= -100
    else: 
        percent *= 100
    if percent > 100 or percent < -100:
        percent /= 2
    print(format(percent, '.2f'), "%")

    analyse = input("Enter your sarcasm: ")
