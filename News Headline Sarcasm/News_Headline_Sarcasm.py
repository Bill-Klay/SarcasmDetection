import json
import csv
import re
from nltk.util import ngrams
from textblob import TextBlob

def parse_data(file):
    for l in open(file,'r'):
        yield json.loads(l)

data = list(parse_data('./Sarcasm_Headlines_Dataset.json'))

unigram_sum = [0] * len(data)
bigram_sum = [0] * len(data)
trigram_sum = [0] * len(data)
sentiment = [0] * len(data)
pos_high = [0] * len(data)
pos_med = [0] * len(data)
pos_low = [0] * len(data)
neg_high = [0] * len(data)
neg_med = [0] * len(data)
neg_low = [0] * len(data)
is_sarcastic = [0] * len(data)

for x in range(len(data)):
    total_sum = 0

    #print(data[x]["headline"])
    analyse = str(data[x]["headline"])

    #blob = TextBlob(analyse)
    #print(blob.sentiment)

    analyse = analyse.lower()
    analyse = re.sub(r'[^a-zA-Z0-9\s]', ' ', analyse)
    tokens = [token for token in analyse.split(" ") if token != ""]
    output = list(ngrams(tokens, 1))
    #unigram_sum[x] = bigram_sum [x] = trigram_sum[x] = 0
    
    for z in range(len(output)):
        #print(output[z])
        blob = TextBlob(str(output[z]))
        #print(blob.sentiment)
        unigram_sum[x] += blob.sentiment.polarity

    output = list(ngrams(tokens, 2))
    for z in range(len(output)):
        #print(output[z])
        blob = TextBlob(str(output[z]))
        #print(blob.sentiment)
        bigram_sum[x] += blob.sentiment.polarity

    output = list(ngrams(tokens, 3))
    for z in range(len(output)):
        #print(output[z])
        blob = TextBlob(str(output[z]))
        #print(blob.sentiment)
        trigram_sum[x] += blob.sentiment.polarity
    
    blob = TextBlob(str(data[x]["headline"]))
    sentiment[x] += blob.sentiment.polarity
    
    is_sarcastic[x] = data[x]["is_sarcastic"]
    total_sum += unigram_sum[x] + bigram_sum[x] + trigram_sum[x]
    if total_sum <= -1:
        pos_low[x] = 1
        neg_high[x] = 1
    elif total_sum >= 0 and total_sum <= 1:
        pos_med[x] = 1
        neg_med[x] = 1
    elif total_sum >= 2:            
        pos_high[x] = 1
        neg_low[x] = 1

    if x % 100 == 0:
        print('Yes its working ', x)

rows = zip(unigram_sum, bigram_sum, trigram_sum, sentiment, is_sarcastic)
with open('With_Sentiment.csv', "w", newline = '') as f:
    writer = csv.writer(f)
    for row in rows:
        writer.writerow(row)

rows = zip(pos_high, pos_med, pos_low, neg_high, neg_med, neg_low, is_sarcastic)
with open('SVM_data.csv', "w", newline = '') as f:
    writer = csv.writer(f)
    for row in rows:
        writer.writerow(row)

print()
for x in range(len(data)):
    print()
    print(format(unigram_sum[x],'.2f'), end=" ")
    print(format(bigram_sum[x],'.2f'), end=" ")
    print(format(trigram_sum[x],'.2f'), end =" ")
    print(data[x]["is_sarcastic"])
    if x % 10 == 0:
        input("Press ENTER to terminate")

