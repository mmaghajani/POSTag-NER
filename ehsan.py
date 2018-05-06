# set the STANFORD_MODELS as you did # I learnt from you, thx!
import nltk.tag.stanford as st
from sklearn.metrics import accuracy_score
import csv
import pandas as pd
from tqdm import tqdm

with open('data/NERte.txt') as f:
    train = f.read().splitlines()
print('loaded')

data = []
for item in train:
    data.append((item.split('\t')))

with open("test.csv", "w+") as my_csv:
    csvWriter = csv.writer(my_csv, delimiter=',')
    csvWriter.writerow(['word', 'tag', ' '])
    csvWriter.writerows(data)

test = pd.read_csv('test.csv')
test = test.drop(' ', axis=1)
# print(test.shape)

tagger = st.StanfordNERTagger('/home/mma137421/stanford-ner/ner-model.ser.gz',
                              '/home/mma137421/stanford-ner/stanford-ner.jar')

x_test = test['word'].tolist()
y_test = test['tag'].tolist()
print(len(x_test))
# print(x_test)

sentence = list()
sentences = list()
output = list()
x = list()
for xx in x_test:
    x.append(str(xx))
print(x)
print(type(x))
predict = tagger.tag(x)
print(len(predict))
# print(y_test)
cnt = 0
prediction = []
for item in predict:
    prediction.append(item[1])

for i in range(len(prediction)):
    if prediction[i] == y_test[i] and y_test[i] != 'O':
        cnt += 1

# print(cnt)
# print(cnt / len(predict))
#
# print(accuracy_score(y_test, prediction))


def boundryMatch(predict, actual):
    length = 0
    counter = 0
    i = 0
    for i in range(len(predict)):

        if predict[i] != 'O':
            length += 1
        else:
            if length != 0:
                if actual[i - length] != 'O' and actual[i - 1] != 'O':
                    counter += 1
                length = 0
    i += 1
    if length != 0:
        if actual[i - length] != 'O' and actual[i - 1] != 'O':
            counter += 1

    counter1 = 0
    for j in range(len(actual)):
        if actual[j] != 'O':
            if j + 1 < len(actual):
                if actual[j + 1] != 'O':
                    continue
                else:
                    counter1 += 1
            else:
                counter1 += 1
    print(counter)
    print(counter1)

    return counter / counter1


def typeMatch(predict, actual):
    counter1 = 0
    for j in range(len(actual)):
        if actual[j] == 'O':
            if j + 1 < len(actual):
                if actual[j + 1] != 'O':
                    if actual[j+1] == predict[j+1]:
                        counter1 += 1
    return counter1




print(boundryMatch(prediction,y_test))
print(typeMatch(prediction,y_test))
